#![allow(dead_code)]

use futures_util::StreamExt;

pub mod chat;
pub mod completions;

pub struct Client {
    client: reqwest::Client,
}

#[derive(serde::Serialize)]
struct WrappedRequest<'a, T> {
    stream: bool,

    #[serde(flatten)]
    req: &'a T,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct StreamError {
    pub error: String,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("request: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("request: {0} ({1})")]
    ReqwestWithBody(reqwest::Error, String),

    #[error("serde: {0}")]
    SerdeJson(#[from] serde_json::Error),

    #[error("stream: {0}")]
    Stream(String),

    #[error("malformed stream item")]
    MalformedStreamItem(Vec<u8>),
}

fn into_sse_stream(mut resp: reqwest::Response) -> impl futures_core::stream::Stream<Item = Result<Vec<u8>, Error>> {
    let mut buf = bytes::BytesMut::new();

    async_stream::try_stream! {
        while let Some(c) = resp.chunk().await.map_err(|e| e.without_url())? {
            buf.extend_from_slice(&c);

            while let Some(i) = buf.windows(2).position(|x| x == b"\n\n") {
                let payload = buf.split_to(i + 2);
                let payload = &payload[..payload.len() - 2];

                if !payload.starts_with(b"data: ") {
                    Err(Error::MalformedStreamItem(payload.to_vec()))?;
                }

                let payload = &payload[6..];
                yield payload.to_vec();
            }
        }
    }
}

impl Client {
    pub fn new(api_key: impl AsRef<str>) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(reqwest::header::AUTHORIZATION, format!("Bearer {}", api_key.as_ref()).parse().unwrap());
        Self {
            client: reqwest::ClientBuilder::new().default_headers(headers).build().unwrap(),
        }
    }

    pub async fn do_streaming_request<Req, Chunk>(
        &self,
        url: &str,
        req: &Req,
    ) -> Result<impl futures_core::stream::Stream<Item = Result<Chunk, Error>>, Error>
    where
        Req: serde::Serialize,
        Chunk: serde::de::DeserializeOwned,
    {
        let resp = self
            .client
            .post(url)
            .json(&WrappedRequest { stream: true, req })
            .send()
            .await
            .map_err(|e| e.without_url())?;

        if let Err(e) = resp.error_for_status_ref() {
            let body = resp.text().await.map_err(|e| e.without_url())?;
            return Err(Error::ReqwestWithBody(e.without_url(), body));
        }

        Ok(async_stream::try_stream! {
            let mut stream = Box::pin(into_sse_stream(resp));

            while let Some(payload) = stream.next().await {
                let payload = payload?;

                if payload == b"[DONE]" {
                    break;
                }

                // Check if there is an error first.
                if let Ok(stream_error) = serde_json::from_slice::<StreamError>(&payload) {
                    Err(Error::Stream(stream_error.error))?;
                }

                yield serde_json::from_slice::<Chunk>(&payload)?;
            }
        })
    }

    pub async fn create_chat_completion(
        &self,
        req: &chat::completions::CreateRequest,
    ) -> Result<impl futures_core::stream::Stream<Item = Result<chat::completions::Chunk, Error>>, Error> {
        Ok(self.do_streaming_request("https://api.openai.com/v1/chat/completions", req).await?)
    }

    pub async fn create_completion(
        &self,
        req: &completions::CreateRequest,
    ) -> Result<impl futures_core::stream::Stream<Item = Result<completions::Chunk, Error>>, Error> {
        Ok(self.do_streaming_request("https://api.openai.com/v1/completions", req).await?)
    }
}
