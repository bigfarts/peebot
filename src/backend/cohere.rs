pub struct Backend {
    client: reqwest::Client,
    model: String,
    max_total_tokens: u32,
    tokenizer: tiktoken_rs::CoreBPE,
}

#[derive(serde::Deserialize)]
pub struct Config {
    model: String,
    api_key: String,
    max_total_tokens: u32,
}

fn convert_message(message: &super::Message) -> String {
    let mut buf = String::new();
    buf.push_str(match message.name.as_ref() {
        Some(name) => &name,
        None => match message.role {
            super::Role::System => "system",
            super::Role::Assistant => "assistant",
            super::Role::User(..) => "user",
        },
    });
    buf.push_str(": ");
    buf.push_str(&message.content);
    buf.push_str("\n");
    buf
}

impl Backend {
    pub fn new(config: &Config) -> Result<Self, anyhow::Error> {
        Ok(Self {
            client: reqwest::ClientBuilder::new()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(reqwest::header::ACCEPT, "application/json".parse().unwrap());
                    headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse().unwrap());
                    headers.insert(reqwest::header::AUTHORIZATION, format!("Bearer {}", config.api_key).parse().unwrap());
                    headers
                })
                .build()
                .unwrap(),
            model: config.model.clone(),
            max_total_tokens: config.max_total_tokens,
            tokenizer: tiktoken_rs::CoreBPE::new(
                serde_json::from_slice::<std::collections::HashMap<String, usize>>(include_bytes!("cohere/coheretext-50k.json"))?
                    .into_iter()
                    .map(|(k, v)| (k.as_bytes().to_vec(), v))
                    .collect(),
                std::collections::HashMap::default(),
                r"(?:'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)",
            )?,
        })
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    pub temperature: Option<f64>,
    pub k: Option<u32>,
    pub p: Option<u32>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
}

#[derive(serde::Serialize)]
struct Request {
    prompt: String,
    model: String,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    k: Option<u32>,
    p: Option<u32>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
}

#[derive(serde::Deserialize)]
struct ResponseGeneration {
    text: String,
}

#[derive(serde::Deserialize)]
struct Response {
    generations: Vec<ResponseGeneration>,
}

#[async_trait::async_trait]
impl super::Backend for Backend {
    async fn request(
        &self,
        messages: &[super::Message],
        parameters: &toml::Value,
    ) -> Result<std::pin::Pin<Box<dyn futures_core::stream::Stream<Item = Result<String, anyhow::Error>> + Send>>, anyhow::Error> {
        let parameters: Parameters = parameters.clone().try_into()?;

        let req = Request {
            prompt: format!("{}assistant:", messages.iter().map(|m| convert_message(m)).collect::<Vec<_>>().join("")),
            model: self.model.clone(),
            temperature: parameters.temperature,
            k: parameters.k,
            p: parameters.p,
            frequency_penalty: parameters.frequency_penalty,
            presence_penalty: parameters.presence_penalty,
            max_tokens: Some(
                self.max_total_tokens - (self.num_overhead_tokens() + messages.iter().map(|m| self.count_message_tokens(m)).sum::<usize>()) as u32,
            ),
        };

        let resp = self
            .client
            .post("https://api.cohere.ai/v1/generate")
            .json(&req)
            .send()
            .await
            .map_err(|e| e.without_url())?;

        if let Err(e) = resp.error_for_status_ref() {
            let body = resp.text().await.map_err(|e| e.without_url())?;
            return Err(anyhow::format_err!("{:?} ({:?})", e.without_url(), body));
        }

        Ok(Box::pin(async_stream::try_stream! {
            yield resp
                .json::<Response>()
                .await
                .map_err(|e| e.without_url())?
                .generations
                .first()
                .ok_or_else(|| anyhow::anyhow!("no generation"))?
                .text
                .clone();
        }))
    }

    fn count_message_tokens(&self, message: &super::Message) -> usize {
        self.tokenizer.encode_ordinary(&convert_message(message)).len()
    }

    fn num_overhead_tokens(&self) -> usize {
        self.tokenizer.encode_ordinary("assistant:").len()
    }

    fn request_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(2 * 60)
    }

    fn chunk_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(2 * 60)
    }
}
