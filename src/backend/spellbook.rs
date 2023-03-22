pub struct Backend {
    client: reqwest::Client,
    deployment_url: String,
    tokenizer: tiktoken_rs::CoreBPE,
}

fn convert_message(message: &super::Message) -> String {
    let mut buf = String::new();
    buf.push_str("<|im_start|>");
    buf.push_str(match message.name.as_ref() {
        Some(name) => &name,
        None => match message.role {
            super::Role::System => "system",
            super::Role::Assistant => "assistant",
            super::Role::User => "user",
        },
    });
    buf.push_str("\n");
    buf.push_str(&message.content);
    buf.push_str("<|im_end|>\n");
    buf
}

impl Backend {
    pub fn new(api_key: impl AsRef<str>, deployment_url: String, tokenizer: tiktoken_rs::CoreBPE) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(reqwest::header::AUTHORIZATION, format!("Basic {}", api_key.as_ref()).parse().unwrap());
        Self {
            client: reqwest::ClientBuilder::new().default_headers(headers).build().unwrap(),
            deployment_url,
            tokenizer,
        }
    }
}

#[derive(serde::Serialize)]
struct RequestInput {
    input: String,
}

#[derive(serde::Serialize)]
struct Request {
    input: RequestInput,
}

#[derive(serde::Deserialize)]
struct Response {
    output: String,
}

#[async_trait::async_trait]
impl super::Backend for Backend {
    async fn request(
        &self,
        req: &super::Request,
    ) -> Result<std::pin::Pin<Box<dyn futures_core::stream::Stream<Item = Result<String, anyhow::Error>> + Send>>, anyhow::Error> {
        let req = Request {
            input: RequestInput {
                input: format!(
                    "{}<|im_start|>assistant",
                    req.messages.iter().map(|m| convert_message(m)).collect::<Vec<_>>().join("")
                ),
            },
        };

        let resp = self.client.post(&self.deployment_url).json(&req).send().await?;

        if let Err(e) = resp.error_for_status_ref() {
            let body = resp.text().await?;
            return Err(anyhow::format_err!("{:?} ({:?})", e, body));
        }

        let resp = resp.json::<Response>().await?;

        Ok(Box::pin(async_stream::try_stream! {
            yield resp.output;
        }))
    }

    fn count_message_tokens(&self, message: &super::Message) -> usize {
        self.tokenizer.encode_ordinary(&convert_message(message)).len()
    }

    fn num_overhead_tokens(&self) -> usize {
        self.tokenizer.encode_ordinary("<|im_start|>assistant").len()
    }

    fn request_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(2 * 60)
    }

    fn chunk_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(2 * 60)
    }
}
