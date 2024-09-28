use anyhow::Result;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use ollama_rs::generation::options::GenerationOptions;
use ollama_rs::Ollama;
use rand::prelude::SliceRandom;

use super::StateMachine;

pub struct OllamaDriver {
    client: Ollama,
    wordlist: Vec<&'static str>,
}

impl OllamaDriver {
    pub fn new(wordlist: Vec<&'static str>) -> Self {
        OllamaDriver {
            client: Ollama::default(),
            wordlist,
        }
    }
}

#[async_trait::async_trait]
impl StateMachine for OllamaDriver {
    fn generate(&self) -> String {
        let mut rng = rand::thread_rng();
        let word = self.wordlist.choose(&mut rng).unwrap();

        word.to_string()
    }

    async fn process(&self, input: String, logit: String) -> Result<(f64, String)> {
        let model = "llama3.2:latest";
        let prompt = format!(
            "Given a word, how well does the following sentence explains what it means?\n\nword: \"{}\"\n\nTell me 3 things:\n1. How well does the sentence describe the word?\n2. Word meaning\n3. Example sentence with the word\n\nThe output should be less than 100 characters.\n\nsentence: \"{}\"", input, logit
        );

        let stmt = self.client.generate(
            GenerationRequest::new(model.to_string(), prompt)
                .options(GenerationOptions::default().temperature(0.5)),
        );

        // let model = "all-minilm";

        let score = self
            .client
            .generate_embeddings(GenerateEmbeddingsRequest::new(
                model.to_string(),
                ollama_rs::generation::embeddings::request::EmbeddingsInput::Multiple(vec![
                    input, logit,
                ]),
            ));

        let (help, score) = tokio::try_join!(stmt, score)?;

        let help = help.response;

        let one = score.embeddings[0].clone();
        let two = score.embeddings[1].clone();

        let score = cosine_similarity(&one, &two);

        Ok((score as f64, help))
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}
