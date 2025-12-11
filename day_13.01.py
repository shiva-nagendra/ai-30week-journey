# hangman OOP v4
#practice 2

#Hangman V3 – Playable Core
#build and show current word state
#import, functions
import random

class Hangman:

   def __init__(self, word, difficulty=None): 
      self.word = word
      self.guessed = []
      self.attempts = max(5, len(self.word) // 2 + 3) 
      self.wrong_guessed = []
      self.difficulty = (difficulty or "").lower()
      self.apply_difficulty()

   def apply_difficulty(self):
       if self.difficulty == "easy":
          print("\nyou have picked EASY mode")
          self.attempts = len(self.word) + 2
       elif self.difficulty == "medium":
          print("\nyou have picked MEDIUM mode")
          self.attempts = len(self.word) 
       elif self.difficulty == "hard":
          print("\nyou have picked HARD mode")
          self.attempts = max(1, len(self.word) - 1)
       else:
          self.attempts = max(5, len(self.word) // 2 + 3) 
        
   def display_word(self):
      display = ""
      for char in self.word:
         if char in self.guessed:
            display += char + " "
         else:
            display += "_ "
      print("Word :", display)

      if self.guessed:
            print("\nguessed so far :" , ", ".join(sorted(self.guessed)))

   def guess_letter(self, letter):
      letter = letter.lower().strip()

      if len(letter) != 1 or not letter.isalpha():
         print("\nOnly enter single alphabet!")
         return False
      
      if letter in self.guessed or letter in self.wrong_guessed:
         print("\nYou have already guessed that letter")
         return False

      if letter in self.word:
         self.guessed.append(letter)
         print("\nGood guess!")
         return True
      else :
        self.wrong_guessed.append(letter)
        self.attempts -= 1
        print ("\nwrong guess! try again...")
        return False
      
   def is_complete(self):
      return all( char in self.guessed for char in self.word)
   
   def remaining_letters(self):
       return sum(1 for ch in set(self.word) if ch not in self.guessed)
      
   def play(self):
      while self.attempts > 0:
         self.display_word()

         print("\nwrong guesses:", ", ".join(self.wrong_guessed) or "None")
         print("\nletters remaining: ", self.remaining_letters())

         print(f"Attempts left: ",self.attempts)
         guess = input("Enter a letter: ")
         self.guess_letter(guess)
        
         if self.is_complete():
             print(f"\nhurrah!! you have won the game.\nYou have guessed it correctly: {self.word}")
             break
         
      if not self.is_complete():
            print(f"\nGame over! The word was: {self.word}")
     
      
if __name__ == "__main__":

    difficulty = input("\n\tChoose difficulty (easy/medium/hard): ").lower()
    word = random.choice(["python", "variable", "function", "loop", "hangman", "engineer"])
    print(f"\nI picked a word with {len(word)} letters.")
    print("\nLet's see if your logic matches mine!")
    game = Hangman(word, difficulty)
    game.play()
    
    



        



    
 
