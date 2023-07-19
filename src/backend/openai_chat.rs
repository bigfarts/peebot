use futures_util::StreamExt;

pub struct Backend {
    client: crate::openai::Client,
    model: String,
    max_total_tokens: u32,
}

#[derive(serde::Deserialize)]
pub struct Config {
    api_key: String,
    model: String,
    max_total_tokens: u32,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
}

impl Backend {
    pub fn new(config: &Config) -> Result<Self, anyhow::Error> {
        Ok(Self {
            client: crate::openai::Client::new(config.api_key.clone()),
            model: config.model.clone(),
            max_total_tokens: config.max_total_tokens,
        })
    }
}

fn convert_message(m: &super::Message) -> crate::openai::chat::completions::Message {
    crate::openai::chat::completions::Message {
        content: m.content.clone(),
        name: m.name.clone(),
        role: match m.role {
            super::Role::System => crate::openai::chat::completions::Role::System,
            super::Role::Assistant => crate::openai::chat::completions::Role::Assistant,
            super::Role::User(..) => crate::openai::chat::completions::Role::User,
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

        let req = {
            let mut req = crate::openai::chat::completions::CreateRequest::new(self.model.clone(), messages.iter().map(convert_message).collect());
            req.temperature = parameters.temperature;
            req.top_p = parameters.top_p;
            req.frequency_penalty = parameters.frequency_penalty;
            req.presence_penalty = parameters.presence_penalty;
            req.max_tokens = Some(
                self.max_total_tokens - (self.num_overhead_tokens() + messages.iter().map(|m| self.count_message_tokens(m)).sum::<usize>()) as u32,
            );
            req
        };
        log::info!("openai request: {:?}", req);

        let mut stream = Box::pin(self.client.create_chat_completion(&req).await?);
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
        tiktoken_rs::num_tokens_from_messages(
            &self.model,
            &[tiktoken_rs::ChatCompletionRequestMessage {
                role: serde_plain::to_string(&match message.role {
                    super::Role::System => crate::openai::chat::completions::Role::System,
                    super::Role::Assistant => crate::openai::chat::completions::Role::Assistant,
                    super::Role::User(..) => crate::openai::chat::completions::Role::User,
                })
                .unwrap(),
                content: Some(message.content.clone()),
                name: message.name.clone(),
                function_call: None,
            }],
        )
        .unwrap_or(0)
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
