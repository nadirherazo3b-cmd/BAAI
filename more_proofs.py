"""
for i in range(1,11):
    print(f"Count: {i},", end=" ")
    if i < 5:
        print(f"Count: {i}")
    else:
        print(f"Count: {i}")
"""

order_values = [120, 450, 80, 300, 650]
total_revenue = 0
count = 0
for order in order_values:
    total_revenue += order
    count += 1
    print(f"Processed order: ${order}, Total revenue so far: ${total_revenue}"
          )
print(f"Final total revenue: ${total_revenue}")
print(f"total items processed: {count}")