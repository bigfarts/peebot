pub mod openai_chat;
pub mod spellbook;

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

#[async_trait::async_trait]
pub trait Backend {
    async fn request(
        &self,
        messages: &[Message],
        parameters: &toml::Value,
    ) -> Result<std::pin::Pin<Box<dyn futures_core::stream::Stream<Item = Result<String, anyhow::Error>> + Send>>, anyhow::Error>;
    fn count_message_tokens(&self, message: &Message) -> usize;
    fn num_overhead_tokens(&self) -> usize;
    fn request_timeout(&self) -> std::time::Duration;
    fn chunk_timeout(&self) -> std::time::Duration;
}

pub fn new_backend_from_config(typ: String, config: toml::Value) -> Result<Box<dyn Backend + Send + Sync>, anyhow::Error> {
    Ok(match typ.as_str() {
        "openai_chat" => {
            let config = config.try_into()?;
            Box::new(openai_chat::Backend::new(&config)?)
        }
        "spellbook" => {
            let config = config.try_into()?;
            Box::new(spellbook::Backend::new(&config)?)
        }
        _ => {
            return Err(anyhow::format_err!("unknown backend type: {}", typ));
        }
    })
}
