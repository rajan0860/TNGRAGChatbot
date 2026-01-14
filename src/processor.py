import os
import re

def strip_parentheses(s):
    return re.sub(r'\(.*?\)', '', s)

def is_single_word_all_caps(s):
    # First, we split the string into words
    words = s.split()

    # Check if the string contains only a single word
    if len(words) != 1:
        return False

    # Make sure it isn't a line number
    if bool(re.search(r'\d', words[0])):
        return False

    # Check if the single word is in all caps
    return words[0].isupper()

def extract_character_lines(file_path, character_name, dialogues_list):
    """
    Extracts lines for a specific character from a script file and appends them to the provided list.
    """
    lines = []
    with open(file_path, 'r') as script_file:
        try:
          lines = script_file.readlines()
        except UnicodeDecodeError:
          pass

    is_character_line = False
    current_line = ''
    current_character = ''
    for line in lines:
        strippedLine = line.strip()
        if (is_single_word_all_caps(strippedLine)):
            is_character_line = True
            current_character = strippedLine
        elif (line.strip() == '') and is_character_line:
            is_character_line = False
            dialog_line = strip_parentheses(current_line).strip()
            dialog_line = dialog_line.replace('"', "'")
            if (current_character == character_name and len(dialog_line)>0):
                dialogues_list.append(dialog_line)
            current_line = ''
        elif is_character_line:
            current_line += line.strip() + ' '

def process_directory(directory_path, character_name):
    """
    Processes all files in the directory and returns a list of dialogues for the character.
    """
    dialogues = []
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist.")
        return dialogues
        
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):  # Ignore directories
            extract_character_lines(file_path, character_name, dialogues)
            
    return dialogues

def save_dialogues(dialogues, output_path):
    """
    Saves the dialogues list to a file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w+") as f:
        for line in dialogues:
            f.write(line + "\n")
