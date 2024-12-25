# Vocab TUI

A powerful Terminal User Interface (TUI) application designed to enhance your vocabulary learning experience. Whether you're preparing for standardized tests like GRE, SAT, or simply wanting to expand your vocabulary, Vocab TUI provides an interactive and efficient way to practice and learn new words.

## Description

Vocab TUI is a Rust-based terminal application that combines the power of Large Language Models with a clean, intuitive interface to help you master vocabulary. The application features:

- Interactive terminal interface for efficient learning
- Real-time feedback on your word usage
- AI-powered sentence evaluation using Ollama
- Comprehensive word list targeting standardized test preparation
- Score tracking and detailed explanations for improvement

## Installation

### Prerequisites

1. **Rust Development Environment**
   - Install Rust using [rustup](https://rustup.rs/)
   - Ensure you have Cargo (Rust's package manager) installed

2. **Ollama Setup**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull required models
   ollama pull llama3.2
   ollama pull all-minilm
   ```

### Building from Source

1. Clone the repository
   ```bash
   git clone https://github.com/NishantJoshi00/vocab-tui.git
   cd vocab-tui
   ```

2. Build and run the application
   ```bash
   cargo build --release
   cargo run --release
   ```

## Usage

1. Start the application:
   ```bash
   cargo run --release
   ```

2. Interface Navigation:
   - `Tab`: Get a new word
   - `Enter`: Submit your sentence for evaluation
   - `Ctrl+R`: Retry current word
   - `Esc` or `Ctrl+C`: Exit application

3. Learning Flow:
   - A word will be displayed in the top box
   - Type a sentence using the word in the input box
   - Press Enter to get AI feedback on your usage
   - Review your score and the explanation
   - Press Tab to move to the next word

## Features

- **Interactive TUI**: Clean and responsive terminal user interface built with Ratatui
- **AI-Powered Feedback**: Real-time sentence evaluation using Ollama's language models
- **Comprehensive Word List**: Carefully curated vocabulary targeting standardized tests
- **Scoring System**: Get immediate feedback on how well you've used the word
- **Detailed Explanations**: Receive AI-generated explanations to improve your understanding
- **Progress Tracking**: Monitor your performance with a scoring system

## Contributing Guidelines

1. Please create an issue before starting work on a new feature or bug fix
2. Do not work on issues that are already assigned
3. Tag your issues appropriately using:
   - [BUG] for bug reports
   - [FEATURE] for feature requests
   - [DOCS] for documentation improvements
4. Ensure your code follows the project's style guidelines
5. Include tests for new features
6. Update documentation as needed
