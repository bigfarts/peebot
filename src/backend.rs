pub mod openai_chat;

#[derive(Debug)]
pub enum Role {
    System,
    Assistant,
    User,
}

#[derive(Debug)]
pub struct Message {
    pub role: Role,
    pub name: Option<String>,
    pub content: String,
}

#[derive(Debug)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub max_tokens: Option<u32>,
}

#[async_trait::async_trait]
pub trait Backend {
    async fn request(
        &self,
        req: &Request,
    ) -> Result<std::pin::Pin<Box<dyn futures_core::stream::Stream<Item = Result<String, anyhow::Error>> + Send>>, anyhow::Error>;
    fn count_message_tokens(&self, message: &Message) -> usize;
}
