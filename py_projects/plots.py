#data visualization 
#plotting on graph

import matplotlib.pyplot as plt

#Days vs Temperature

days = [1,2,3,4,5,6,7]
temperature = [30,32,32,31,30,34,36]

plt.figure()
plt.plot(days,temperature)
plt.title("Temperature vs Days")
plt.xlabel("Days")
plt.ylabel("Temperature (c*)")
plt.plot(days, temperature, marker = "o", markersize = 5, markerfacecolor = "red")
plt.grid(True)
plt.show()

#Bar chart: sales per month

months = ["jan","feb","mar","apr"]
sales = [200, 243, 260, 280]

plt.figure()
plt.bar(months,sales)
plt.title("Quarterly sales report")
plt.xlabel("month")
plt.ylabel("sales")
plt.show()

#Histogram

marks = [34, 45, 54, 55, 67, 88, 98, 99]

plt.figure()
plt.hist(marks, bins=5)
plt.title("Marks histogram")
plt.xlabel("marks")
plt.ylabel("frequency")
plt.show()


#Scatter Plot: Height vs Weight

height = [150, 155, 160, 165, 170, 175, 180]
weight = [50, 55, 60, 65, 70, 75, 80]

plt.figure()
plt.scatter(height, weight)
plt.title("Height vs Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.show()