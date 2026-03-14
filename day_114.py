#week 19 day 2
#text summerizer - transformers

from transformers import pipeline

summerizer = pipeline("summarization",
                      model="sshleifer/distilbart-cnn-6-6",
                      )

text = """
I will explain the process how to book a movie ticket online\
At first we have to open google chrome or safari on your mobile phone\
Then we need to open a website called bookmyshow.
It will open and shows certain options like book an event,book a movie ticket\
book a musical night, book a dinner table.
Now we have to select the right option as book a movie ticket. Then we have to select the movie we\
want to watch and then it will display available seats in that show.
Select the seat numbers. Go to payment mode, make payment.
That's it it will display as You booked your movie tickets with screen no and ticket numbers\
Enjoy the movie.
"""

summerize = summerizer(
    text,
    max_length=30,
    min_length=10,
    do_sample=False)

print("\nsummerized text:",summerize)