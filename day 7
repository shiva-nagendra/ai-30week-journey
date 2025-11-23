# Write a program that:

# Continuously asks the user to enter words.

# Store all entered words in a list (to preserve order + duplicates allowed).

# Also keep track of unique words only using a set.

# When the user types "stop", exit input.

# Then print:

# All words entered (list)

# Unique words (set)

# Total words entered

# Total unique words

# Which word was entered most often

All_words = []
while True:
    words = input("Enter a word or \"exit\" to the list: ").lower()
    if words == "exit":
        break

    All_words.append(words)

unique_words = set(All_words)

print("total words are: ", All_words)
print("total unique words are: ",unique_words)
print("no. of all words: ",len(All_words))
print("no.of unique words: ",len(unique_words))
print("most repeated word is: ",max(All_words, key = All_words.count))
    



        




    

