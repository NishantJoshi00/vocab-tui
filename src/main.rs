use anyhow::Result;
use crossterm::event::{self, Event};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::sync::{Arc, RwLock};
use tokio::runtime;

mod ollama_driver;

struct App {
    flow_marker: bool,
    config: Config,
    state: Arc<RwLock<State>>,
    input: String,
    display: String,
    shared_state: Arc<RwLock<SlowState>>,
    state_machine: Arc<dyn StateMachine>,
}

struct Config {
    top_box_name: String,
    left_box_name: String,
    right_top_name: String,
    right_bottom_name: String,
}

enum State {
    Input,
    Processing,
    Review,
}

impl State {
    fn is_input(&self) -> bool {
        matches!(self, State::Input)
    }
    // fn is_processing(&self) -> bool {
    //     matches!(self, State::Processing)
    // }
    // fn is_review(&self) -> bool {
    //     matches!(self, State::Review)
    // }
}

#[async_trait::async_trait]
trait StateMachine: Send + Sync {
    fn generate(&self) -> String;
    async fn process(&self, input: String, logit: String) -> Result<(f64, String)>;
}

struct SlowState {
    explanation: String,
    score: f64,
}

impl App {
    fn new(state_machine: Arc<dyn StateMachine>, config: Config) -> Self {
        App {
            flow_marker: false,
            config,
            state: Arc::new(RwLock::new(State::Input)),
            input: String::new(),
            display: state_machine.generate(),
            shared_state: Arc::new(RwLock::new(SlowState {
                explanation: String::new(),
                score: 0.0,
            })),
            state_machine,
        }
    }

    fn on_next(&mut self) {
        self.input.clear();
        self.display = self.state_machine.generate();
        let mut shared_state = self
            .shared_state
            .write()
            .expect("Failed to lock shared state");
        shared_state.explanation.clear();
        shared_state.score = 0.0;
        drop(shared_state);
        *self.state.write().expect("Failed to lock state") = State::Input;
    }

    fn on_review(&mut self, rt: &runtime::Runtime) {
        if !self.state.read().unwrap().is_input() {
            return;
        }

        let input = self.input.clone();
        let logit = self.display.clone();
        let state_machine = self.state_machine.clone();
        let shared_state = self.shared_state.clone();

        *self.state.write().unwrap() = State::Processing;

        let state = self.state.clone();

        rt.spawn(async move {
            let (score, explanation) = state_machine.process(input, logit).await.unwrap();
            let mut state = state.write().unwrap();
            if let State::Processing = *state {
                let mut shared_state = shared_state.write().unwrap();
                shared_state.score = score;
                shared_state.explanation = explanation;
                *state = State::Review;
            }
        });
    }

    fn on_key(&mut self, c: char) {
        if !self.state.read().unwrap().is_input() {
            return;
        }
        self.input.push(c);
    }

    fn on_retry(&mut self) {
        self.input.clear();
        let mut shared_state = self
            .shared_state
            .write()
            .expect("Failed to lock shared state");
        shared_state.explanation.clear();
        shared_state.score = 0.0;
        drop(shared_state);
        *self.state.write().expect("Failed to lock state") = State::Input;
    }

    fn ui(&mut self, f: &mut Frame) {
        let style = Style::new()
            .bg(Color::Rgb(0, 0, 0))
            .fg(Color::Rgb(255, 255, 255));

        let input_style = {
            let state = self.state.read().unwrap();
            match *state {
                State::Input => style,
                State::Processing => Style::default().bg(Color::Rgb(0, 0, 0)).fg(Color::Yellow),
                State::Review => Style::default().bg(Color::Rgb(0, 0, 0)).fg(Color::Green),
            }
        };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Ratio(2, 5),
                Constraint::Ratio(2, 5),
                Constraint::Length(1),
            ])
            .split(f.area());

        let top_box = Paragraph::new(self.display.as_str())
            .block(Block::bordered().title(self.config.top_box_name.as_str()))
            .style(style)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
        f.render_widget(top_box, chunks[0]);

        let help_line = const { "Esc: Quit | Enter: Evaluate | Tab: Next" };
        let last_line = Paragraph::new(help_line)
            .style(style)
            .alignment(Alignment::Center);
        f.render_widget(last_line, chunks[2]);

        let middle_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
            .split(chunks[1]);

        let input_text = if self.flow_marker {
            self.flow_marker = false;
            Line::from(vec![self.input.as_str().into(), "_".into()])
        } else {
            self.flow_marker = true;
            Line::from(self.input.as_str())
        };

        let input_box = Paragraph::new(input_text)
            .block(
                Block::default()
                    .title(self.config.left_box_name.as_str())
                    .borders(Borders::ALL),
            )
            .style(input_style)
            .wrap(Wrap { trim: true });
        f.render_widget(input_box, middle_chunks[0]);

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Ratio(6, 7), Constraint::Ratio(1, 7)])
            .split(middle_chunks[1]);
        {
            let shared_state = self.shared_state.read().unwrap();
            let right_top_box = Paragraph::new(shared_state.explanation.as_str())
                .block(
                    Block::default()
                        .title(self.config.right_top_name.as_str())
                        .borders(Borders::ALL),
                )
                .style(style)
                .wrap(Wrap { trim: true });
            f.render_widget(right_top_box, right_chunks[0]);
            let right_bottom_box = Paragraph::new(format!("{:.2}", shared_state.score))
                .block(
                    Block::default()
                        .title(self.config.right_bottom_name.as_str())
                        .borders(Borders::ALL),
                )
                .style(style)
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true });
            f.render_widget(right_bottom_box, right_chunks[1]);
        }
    }
}

pub fn main() -> Result<()> {
    let config = Config {
        top_box_name: "Word".to_string(),
        left_box_name: "Input".to_string(),
        right_top_name: "Explanation".to_string(),
        right_bottom_name: "Score".to_string(),
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_time()
        .enable_io()
        .build()
        .expect("Failed to build runtime");

    let words = include_str!("../wordlist.txt");
    let words = words.lines().collect::<Vec<_>>();

    let sm = ollama_driver::OllamaDriver::new(words);

    let mut app = App::new(Arc::new(sm), config);

    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;

    let backend = ratatui::backend::CrosstermBackend::new(stdout);

    let mut terminal = ratatui::Terminal::new(backend)?;

    loop {
        terminal.draw(|f| app.ui(f))?;

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match (key.code, key.modifiers) {
                    (event::KeyCode::Char('c'), event::KeyModifiers::CONTROL) => break,
                    (event::KeyCode::Char('d'), event::KeyModifiers::CONTROL) => break,
                    (event::KeyCode::Char('r'), event::KeyModifiers::CONTROL) => app.on_retry(),
                    (event::KeyCode::Delete, _) => app.input.clear(),
                    (event::KeyCode::Char(c), _) => app.on_key(c),
                    (event::KeyCode::Enter, _) => app.on_review(&runtime),
                    (event::KeyCode::Esc, _) => break,
                    (event::KeyCode::Tab, _) => app.on_next(),
                    (event::KeyCode::Backspace, _) => {
                        if app.state.read().unwrap().is_input() {
                            app.input.pop();
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;

    Ok(())
}
