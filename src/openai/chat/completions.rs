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

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Delta {
    pub role: Option<Role>,
    pub name: Option<String>,
    pub content: Option<String>,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Choice {
    pub delta: Delta,
    pub index: i64,
    pub finish_reason: Option<FinishReason>,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Chunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(serde::Serialize, Default, Clone, Debug)]
pub struct CreateRequest {
    pub model: String,

    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<u32, u32>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}
