    def __getitem__(self, idx):
        formatted = format_sample(self.samples[idx])
        tokenized = self.tokenizer(
            formatted,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze(),  # Ensure labels are included
        }