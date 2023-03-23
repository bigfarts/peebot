#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Choice {
    pub text: String,
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

#[derive(serde::Serialize, Clone, Debug)]
pub struct CreateRequest {
    pub model: String,

    pub prompt: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,

    pub echo: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<u32, u32>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl CreateRequest {
    pub fn new(model: String, prompt: Vec<String>) -> Self {
        Self {
            model,
            prompt,
            suffix: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            logprobs: None,
            echo: false,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
        }
    }
}
