use dialoguer::History;
use std::collections::VecDeque;

// based on https://github.com/console-rs/dialoguer/blob/master/examples/history.rs

pub struct InMemoryHistory {
    max: usize,
    history: VecDeque<String>,
}

impl Default for InMemoryHistory {
    fn default() -> Self {
        Self {
            max: 20,
            history: VecDeque::new(),
        }
    }
}

impl<T: ToString> History<T> for InMemoryHistory {
    fn read(&self, pos: usize) -> Option<String> {
        self.history.get(pos).cloned()
    }

    fn write(&mut self, val: &T) {
        if self.history.len() == self.max {
            self.history.pop_back();
        }
        self.history.push_front(val.to_string());
    }
}
