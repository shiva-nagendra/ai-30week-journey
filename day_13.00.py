# hangman OOP version

#Hangman V3 – Playable Core
#build and show current word state
#import, functions
import random

class Hangman:
   def __init__(self, word): 
      self.word = word
      self.attempts = len(word) + 1
      self.guessed = []


   def display_word(self):
      display = ""
      for char in self.word:
         if char in self.guessed:
            display += char + " "
         else:
            display += "_ "
      print("Word : ", display)

   def guess_letter(self, letter):
      letter = letter.lower().strip()

      if len(letter) != 1 or not letter.isalpha():
         print("Only enter single alphabet!")
         return False
      if letter in self.guessed:
         print("You have already guessed that letter")
         return False
      self.guessed.append(letter)

      if letter in self.word:
         print("Good guess!")
         return True
      else: 
        self.attempts -= 1
        print ("wrong guess! try again...")
        return False
   def is_complete(self):
      return all( char in self.guessed for char in self.word)
   

   def play(self):
      while self.attempts > 0:
         self.display_word()
         guess = input("Enter a letter: ")
         self.guess_letter(guess)
         print(f"Attempts left: {self.attempts}")
         
         if self.is_complete():
            print("hurrah!! you have won the game")
            break
      if not self.is_complete():
            print(f"Game over! The word was: {self.word}")

      
      
      
if __name__ == "__main__":
    word = random.choice(["python", "variable", "function", "loop", "hangman", "engineer"])
    game = Hangman(word)
    game.play()



        



    
 

        



    
 
