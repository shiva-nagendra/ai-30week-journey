#Hangman V3 – Playable Core
#build and show current word state
#import, functions

import os
import time
import random

def clear_screen():
   
   """clear the terminal screen(works on windows, mac, linux)"""
   os.system('cls' if os.name == 'nt' else 'clear')

def type_effect(text, delay = 0.03):
    """print text with small delay in characters"""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

WORDS = ["python", "variable", "function", "loop", "hangman", "engineer"]

def play_hangman():
    """play one round of hangman"""
    word = random.choice(WORDS)
    attempts = max(5, len(word) + 1)
    guessed = []

    total_guesses = 0
    correct_guesses = 0
    wrong_guesses = 0

    while attempts > 0:
        clear_screen()
        print(f"\nattempts left: {attempts}")

    #build and show current word state
        display_word = ""
        for char in word:  
          if char in guessed:
             display_word += char + " "

          else:
            display_word += "_ "
        print("Word:", display_word)


    #win condition

        if all(char in guessed for char in word):
         type_effect("\nYou guessed the word! you won")
         break
            

        print(f"guessed so far: {', '.join(guessed) if guessed else 'none'}")

        guess = input("enter a letter: ").lower().strip()

        #validation
    
        if len(guess) != 1 or not guess.isalpha():
           print("type only single alphabet! ")
           time.sleep(1)
           continue

    #correct/wrong logic

        if guess in guessed:
           print(f"guessed already....")
           time.sleep(1)
           continue

        total_guesses += 1

        if guess in word:
            guessed.append(guess)
            correct_guesses += 1
            print("good guess!")

        else: 
            attempts -= 1
            wrong_guesses += 1
            print("wrong guess")
        time.sleep(1)

    else:
       type_effect(f"game is over! the word is: '{word}'")

    print("\n game summary: ")
    print(f"word was.          : {word}")  
    print(f"total guesses      : {total_guesses}")
    print(f"correct guesses.   : {correct_guesses}")

    print(f"wrong guesses.     : {wrong_guesses}")

def main():
   type_effect("welcome to hangman_v3!")
   while True:
      play_hangman()
      choice = input("\n play again Y/N? ").lower().strip()
      if choice != "y":
         print("Thanks for playing!!")
         break
      
if __name__ == "__main__":
    main()         







          

   

    

    
    

    

    



   



