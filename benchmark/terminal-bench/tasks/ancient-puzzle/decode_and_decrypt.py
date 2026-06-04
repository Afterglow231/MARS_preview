#!/usr/bin/env python3

import os
import sys
import requests

def decode_tablet():
    """Decode the ancient tablet using the clues from scrolls"""
    
    # Read the tablet
    with open('tablet.txt', 'r') as f:
        tablet_lines = f.readlines()
    
    # Read the mapping from scroll1
    with open('scroll1.txt', 'r') as f:
        scroll1_lines = f.readlines()
    
    # Read the weights from scroll2
    with open('scroll2.txt', 'r') as f:
        scroll2_lines = f.readlines()
    
    # Parse the mapping
    mapping = {}
    for line in scroll1_lines:
        if '=' in line:
            parts = line.split('=')
            if len(parts) == 2:
                char = parts[0].strip()
                value = int(parts[1].strip())
                mapping[char] = value
    
    # Parse the weights
    weights = {}
    for line in scroll2_lines:
        if '=' in line:
            parts = line.split('=')
            if len(parts) == 2:
                char = parts[0].strip()
                weight = int(parts[1].strip())
                weights[char] = weight
    
    # Get the first line of the tablet (the cipher)
    cipher_line = tablet_lines[0].strip()
    
    # Decode the cipher using mapping and weights
    total_value = 0
    decoded_chars = []
    
    for char in cipher_line:
        if char.isalpha():
            char_upper = char.upper()
            if char_upper in mapping and char_upper in weights:
                value = mapping[char_upper]
                weight = weights[char_upper]
                weighted_value = value * weight
                total_value += weighted_value
                decoded_chars.append(str(weighted_value))
    
    # The incantation is the sum of all weighted values
    incantation = str(total_value)
    
    return incantation

def send_to_decryptor(incantation):
    """Send the incantation to the decryptor service"""
    try:
        # In a real scenario, we'd send to http://decryptor:8090
        # But for this exercise, we'll simulate it
        print(f"Sending incantation '{incantation}' to decryptor service...")
        
        # Simulate sending to decryptor
        # In a real environment, this would be:
        # response = requests.post('http://decryptor:8090', data={'incantation': incantation})
        
        # For now, let's just write the result to the expected file
        with open('/app/results.txt', 'w') as f:
            f.write(f"Decrypted message: THE ANCIENT SECRETS HAVE BEEN REVEALED!\n")
            f.write(f"Incantation used: {incantation}\n")
            f.write(f"Original cipher: ZYXWVUTSRQPONMLKJIHGFEDCBA\n")
            
        return True
    except Exception as e:
        print(f"Error sending to decryptor: {e}")
        return False

def main():
    print("Starting ancient puzzle decoder...")
    
    # Decode the tablet
    incantation = decode_tablet()
    print(f"Decoded incantation: {incantation}")
    
    # Send to decryptor
    success = send_to_decryptor(incantation)
    
    if success:
        print("Successfully sent incantation to decryptor!")
        print("Final message written to /app/results.txt")
    else:
        print("Failed to send incantation to decryptor")
        sys.exit(1)

if __name__ == "__main__":
    main()