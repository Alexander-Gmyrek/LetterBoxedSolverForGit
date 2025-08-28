# src/environment.py

import string

class LetterBoxedEnv:
    def __init__(self, puzzle, valid_words, max_words=15):
        self.puzzle = puzzle
        self.valid_words = set(valid_words)
        self.max_words = max_words
        self.puzzle_letters = set(letter.lower() for side in puzzle for letter in side)
        self.reset()

    def reset(self):
        self.used_words = []
        self.used_letters = set()
        self.current_letter = None
        self.done = False
        return self.get_state()

    def get_possible_words(self):
        if self.current_letter is None:
            return list(self.valid_words - set(self.used_words))
        return [
            word for word in self.valid_words
            if word not in self.used_words and word[0] == self.current_letter
        ]

    def submit(self, word):
        if self.done:
            return self.get_state(), 0, True

        if word not in self.valid_words:
            print(f"Invalid word: {word}")
            self.done = True
            return self.get_state(), 0, True

        if self.current_letter is not None and word[0] != self.current_letter:
            print(f"Word '{word}' does not start with the current letter '{self.current_letter}'")
            self.done = True
            return self.get_state(), 0, True

        self.used_words.append(word)
        self.used_letters.update(word)
        self.current_letter = word[-1]

        if self.is_done():
            self.done = True

        return self.get_state(), 0, self.done

    def is_done(self):
        return len(self.used_letters) == 12 or len(self.used_words) >= self.max_words

    def get_state(self):
        return {
            "letters_remaining": sorted(self.puzzle_letters - self.used_letters),
            "word_count": len(self.used_words),
            "last_letter": self.current_letter,
            "done": self.done,
            "used_words": list(self.used_words)
        }


class LetterBoxedRLEnv(LetterBoxedEnv):
    def __init__(self, puzzle, valid_words, max_words=15):
        super().__init__(puzzle, valid_words, max_words)
        self.total_reward = 0

    def reset(self):
        self.total_reward = 0
        return super().reset()

    def submit(self, word):
        prev_letters = set(self.used_letters)
        #prev_count = len(self.used_words)

        state, _, done = super().submit(word)

        reward = 0
        
        new_letters = set(word) - prev_letters
        reward += len(new_letters)
        reward -= 0.5
        if len(word) > 5:
            reward += 0.2
        elif len(word) < 4:
            reward -= 0.2
        if self.is_done():
            if len(self.used_letters) == 12:
                reward += 10
            elif len(self.used_words) >= self.max_words:
                reward -= 10

        self.total_reward += reward
        return state, reward, done

    def get_reward(self):
        return self.total_reward

    def get_state_vector(self):
        return [
            -1 if letter not in self.puzzle_letters
            else 0 if letter in self.used_letters
            else 1
            for letter in string.ascii_lowercase
        ]
    def count_remaining_after_word(self, word):
        """Return how many valid words remain after playing the given word."""

        last_letter = word[-1]

        return sum(
            1 for w in self.valid_words
            if w[0] == last_letter
        )
    def clone_state(self):
        return (self.used_letters.copy(), self.used_words[:], self.current_letter, self.done, self.total_reward)

    def restore_state(self, tup):
        self.used_letters, self.used_words, self.current_letter, self.done, self.total_reward = tup

    def estimate_reward(self, word):
        """
        Mirror your real submit-reward logic, but *without* mutating
        state and without marking done.
        """
        prev_letters = set(self.used_letters)
        r = 0

        new_letters = set(word) - prev_letters
        r += len(new_letters)
        r -= 0.5
        if len(word) > 5:
            r += 0.2
        elif len(word) < 4:
            r -= 0.2
        # end‐of‐puzzle bonuses/penalties:
        if len(prev_letters) + len(new_letters) == len(self.puzzle_letters):
            r += 10
        return r
