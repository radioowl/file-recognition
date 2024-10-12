import os
import json
from audio_processing import AudioProcessor
from command_recognition import CommandRecognizer

def process_folder(folder_path, model_path, commands_json_file, log_file):
    log_data = []
    audio_processor = AudioProcessor()
    command_recognizer = CommandRecognizer(model_path)

    commands = command_recognizer.load_commands_from_json(commands_json_file)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            input_file = os.path.join(folder_path, file_name)
            output_file = os.path.join(folder_path, f"cleaned_{file_name}")

            # Очистка и усиление аудио
            audio_processor.clean_and_amplify_audio(input_file, output_file, low_cutoff=85, high_cutoff=3000, gain=1)

            # Распознавание аудио
            result = command_recognizer.transcribe_audio(output_file, list(commands.keys()))

            # Добавление результата в лог
            log_data.append({
                "file": file_name,
                "result": result
            })
            print(f"Processed {file_name}: {result}")

    # Сохранение логов
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

# Пример использования
folder_path = 'audio_files'
model_path = "model_small"
commands_json_file = 'commands.json'
log_file = 'log.json'

process_folder(folder_path, model_path, commands_json_file, log_file)
