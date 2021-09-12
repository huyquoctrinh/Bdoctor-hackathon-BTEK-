height = int(input())
weight = int(input())
x = (weight/(height*height))
if (x <18.5):
    print("Underweight")
elif ((x >=18.5) & (x <25)):
    print("Normal")
else:
    print("Overweight")