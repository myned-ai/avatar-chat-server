
from google.genai import types
import json

try:
    fields = list(types.AudioTranscriptionConfig.model_fields.keys())
    print("AudioTranscriptionConfig keys:")
    print(json.dumps(fields, indent=2))
except AttributeError:
    # It might be named differently or just be a dict in some versions?
    # but likely it's a Pydantic model
    print("Could not find AudioTranscriptionConfig model fields")
    print(dir(types))

print("-" * 20)
try:
    help(types.AudioTranscriptionConfig)
except:
    print("No help available")
