#list and loops

fruits = ["apple", "banana", "mango"]

for fruit in fruits:
    print(fruit)

for i in range(1, 6):
    print(i)

#to know num of evens between two num

evens = []     #empty list to store even numbers
for i in range(1,50):
    if i % 2 == 0:
        evens.append(i)
print(evens)
print("total num of evens are: ", len(evens))
print("sum of these even nums is: ", sum(evens))

total = 0

for i in range(1, 11):
    total += i

print(total)

#weekend calender loop

days = ['mon','tues','wed', 'sat','sun']
for day in days: 
    if day in ['sat','sun']:
        continue
    print(f'workday : {day}')

#bad email detection

emails = ['shiv@gmail',
         'shiv@outlook',
         'dropdead sql;',
         'dreqad@email'
]
for email in emails:
    print(f'processing email: {email}')
    if ';' in email:
        print('hacker attack')
        break

#check for even numbers

items = range(1, 6)
for i in items:
    if i % 2 == 0:
        print(f'even num detected: {i}')
else:
    print('odd num detected')

#check whether files end with .csv

files = ['report.csv',
        'file.txt',
        'report2.pdf',
        'data.csv']
for file in files:
    if not file.endswith('.csv'):
        print(f'{file} is not a csv')
else:
    print(f'{file}file is csv')



count = 1

while count <= 5:
    print(count)
    count += 1


#while loop practice


attempts = 0
while attempts < 3:
    answer = input("enter your answer (yes/no)? ")
    if answer == "yes":
        print("glad we are on the same page")
        break
    attempts += 1
    
else: 
    print("three strikes you are out!")








