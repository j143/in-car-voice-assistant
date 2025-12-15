"""Generate synthetic in-car dialog data."""

from typing import List, Dict
import random


class SyntheticDataGenerator:
    """Generate synthetic training data for in-car voice assistant."""

    def __init__(
        self,
        num_samples: int = 1000,
        intents: List[str] | None = None,
        model: str = "mistral-7b",
        quantization: str = "4bit",
    ) -> None:
        self.num_samples = num_samples
        self.intents = intents or [
            "music_control",
            "navigation",
            "climate",
            "error_handling",
        ]
        self.model = model
        self.quantization = quantization

    def generate(self) -> List[Dict]:
        """Generate synthetic samples."""
        samples: List[Dict] = []

        templates = {
            "music_control": [
                "Play {artist}",
                "Next song",
                "Pause music",
                "Set volume to {level}",
            ],
            "navigation": [
                "Navigate to {location}",
                "Show me the nearest {poi}",
                "Start route guidance",
            ],
            "climate": [
                "Set temperature to {temp}",
                "Increase fan speed",
                "Turn on heated seats",
            ],
            "error_handling": [
                "Engine warning",
                "Battery low",
                "Tire pressure alert",
            ],
        }

        artists = ["Coldplay", "Adele", "Daft Punk", "Taylor Swift"]
        levels = ["low", "medium", "high", "50%", "80%"]
        locations = ["home", "work", "airport", "downtown"]
        pois = ["gas station", "coffee shop", "hospital", "parking"]
        temps = ["68", "72", "75", "22C", "19C"]

        for i in range(self.num_samples):
            intent = random.choice(self.intents)
            template = random.choice(templates.get(intent, ["unknown"]))

            text = template.format(
                artist=random.choice(artists),
                level=random.choice(levels),
                location=random.choice(locations),
                poi=random.choice(pois),
                temp=random.choice(temps),
            )

            samples.append(
                {
                    "id": i,
                    "intent": intent,
                    "text": text,
                    "audio_path": f"data/synthetic/audio/{i}.wav",
                }
            )

        return samples
