use futures_util::StreamExt;

pub struct Backend {
    client: crate::openai::ChatClient,
    model: String,
    tokenizer: tiktoken_rs::CoreBPE,
    max_total_tokens: u32,
}

#[derive(serde::Deserialize)]
pub struct Config {
    api_key: String,
    model: String,
    max_total_tokens: u32,
}

#[derive(serde::Deserialize)]

struct Parameters {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
}

impl Backend {
    pub fn new(config: &Config) -> Result<Self, anyhow::Error> {
        Ok(Self {
            client: crate::openai::ChatClient::new(config.api_key.clone()),
            model: config.model.clone(),
            tokenizer: if config.model == "gpt-3.5-turbo" {
                tiktoken_rs::cl100k_base()?
            } else {
                return Err(anyhow::anyhow!("unknown model"));
            },
            max_total_tokens: config.max_total_tokens,
        })
    }
}

fn convert_message(m: &super::Message) -> crate::openai::Message {
    crate::openai::Message {
        content: m.content.clone(),
        name: m.name.clone(),
        role: match m.role {
            super::Role::System => crate::openai::Role::System,
            super::Role::Assistant => crate::openai::Role::Assistant,
            super::Role::User(..) => crate::openai::Role::User,
        },
    }
}

#[async_trait::async_trait]
impl super::Backend for Backend {
    async fn request(
        &self,
        messages: &[super::Message],
        parameters: &toml::Value,
    ) -> Result<std::pin::Pin<Box<dyn futures_core::stream::Stream<Item = Result<String, anyhow::Error>> + Send>>, anyhow::Error> {
        let parameters: Parameters = parameters.clone().try_into()?;

        let req = crate::openai::ChatRequest {
            messages: messages.iter().map(convert_message).collect(),
            model: self.model.clone(),
            temperature: parameters.temperature,
            top_p: parameters.top_p,
            frequency_penalty: parameters.frequency_penalty,
            presence_penalty: parameters.presence_penalty,
            max_tokens: Some(
                self.max_total_tokens - (self.num_overhead_tokens() + messages.iter().map(|m| self.count_message_tokens(m)).sum::<usize>()) as u32,
            ),
        };

        let mut stream = Box::pin(self.client.request(&req).await?);
        Ok(Box::pin(async_stream::try_stream! {
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                let delta = &chunk.choices[0].delta;
                let content = if let Some(content) = delta.content.as_ref() {
                    content
                } else {
                    continue;
                };
                yield content.clone();
            }
        }))
    }

    fn count_message_tokens(&self, message: &super::Message) -> usize {
        crate::openai::count_message_tokens(&self.tokenizer, &convert_message(message))
    }

    fn num_overhead_tokens(&self) -> usize {
        2
    }

    fn request_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(30)
    }

    fn chunk_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(30)
    }
}
