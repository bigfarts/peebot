#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    Assistant,
    User,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub content: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Delta {
    pub role: Option<Role>,
    pub name: Option<String>,
    pub content: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Choice {
    pub delta: Delta,
    pub index: i64,
    pub finish_reason: Option<FinishReason>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct StreamError {
    pub error: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Chunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(serde::Serialize, serde::Deserialize, Default, Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

pub struct ChatClient {
    client: reqwest::Client,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("reqwest: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("serde: {0}")]
    SerdeJson(#[from] serde_json::Error),

    #[error("stream: {0}")]
    Stream(String),

    #[error("malformed stream item")]
    MalformedStreamItem(Vec<u8>),
}

impl ChatClient {
    pub fn new(api_key: impl AsRef<str>) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(reqwest::header::AUTHORIZATION, format!("Bearer {}", api_key.as_ref()).parse().unwrap());
        Self {
            client: reqwest::ClientBuilder::new().default_headers(headers).build().unwrap(),
        }
    }

    pub async fn request(&self, req: &ChatRequest) -> Result<impl futures_core::stream::Stream<Item = Result<Chunk, Error>>, reqwest::Error> {
        #[derive(serde::Serialize)]
        struct WrappedRequest<'a> {
            stream: bool,
            #[serde(flatten)]
            req: &'a ChatRequest,
        }

        let mut resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .json(&WrappedRequest { stream: true, req })
            .send()
            .await?
            .error_for_status()?;

        let mut buf = bytes::BytesMut::new();

        Ok(async_stream::try_stream! {
            while let Some(c) = resp.chunk().await? {
                buf.extend_from_slice(&c);

                while let Some(i) = buf.windows(2).position(|x| x == b"\n\n") {
                    let payload = buf.split_to(i + 2);
                    let payload = &payload[..payload.len() - 2];

                    if !payload.starts_with(b"data: ") {
                        Err(Error::MalformedStreamItem(payload.to_vec()))?;
                    }

                    let payload = &payload[6..];
                    if payload == b"[DONE]" {
                        break;
                    }

                    // Check if there is an error first.
                    if let Ok(stream_error) = serde_json::from_slice::<StreamError>(payload) {
                        Err(Error::Stream(stream_error.error))?;
                    }

                    yield serde_json::from_slice::<Chunk>(payload)?;
                }
            }
        })
    }
}

#[allow(dead_code)]
pub fn count_tokens(tokenizer: &tiktoken_rs::CoreBPE, messages: &[Message]) -> usize {
    // every reply is primed with <im_start>assistant
    messages.iter().map(|m| count_message_tokens(tokenizer, m)).sum::<usize>() + 2
}

pub fn count_message_tokens(tokenizer: &tiktoken_rs::CoreBPE, message: &Message) -> usize {
    // every message follows <im_start>{role/name}\n{content}<im_end>\n
    let mut n = 4;
    n += tokenizer.encode_ordinary(&serde_plain::to_string(&message.role).unwrap()).len();
    if let Some(name) = message.name.as_ref() {
        // if there's a name, the role is omitted
        // role is always required and always 1 token
        n -= 1;
        n += tokenizer.encode_ordinary(&name).len();
    }
    n += tokenizer.encode_ordinary(&message.content).len();
    n
}
