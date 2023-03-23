#[derive(serde::Deserialize, Clone, Debug)]
pub struct Result {
    pub categories: std::collections::HashMap<String, bool>,
    pub categories_scores: std::collections::HashMap<String, f64>,
    pub flagged: bool,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct CreateResponse {
    pub id: String,
    pub model: String,
    pub results: Vec<Result>,
}

#[derive(serde::Serialize, Clone, Debug)]
pub struct CreateRequest {
    pub input: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl CreateRequest {
    pub fn new(input: Vec<String>) -> Self {
        Self { input, model: None }
    }
}
