#
# Project, 2025/10/15
# File: Project.py
# This is snippet task
#

# 1. Input
#Data Provided:
products = [
{"name": "Laptop", "price": 1200, "category": "Electronics"},
{"name": "Shirt", "price": 45, "category": "Clothing"},
{"name": "Phone", "price": 800, "category": "Electronics"},
{"name": "Shoes", "price": 120, "category": "Clothing"},
{"name": "Tablet", "price": 350, "category": "Electronics"},
{"name": "Jacket", "price": 95, "category": "Clothing"},
{"name": "Book", "price": 25, "category": "Books"},
{"name": "Headphones", "price": 150, "category": "Electronics"}
]

total_original = 0
total_discount_amount = 0
total_final = 0
product_count = 0

print("PRODUCT DISCOUNT CALCULATOR")

# 2. Process
#Loop
for product in products:
    name = product["name"]
    category = product["category"]
    price = product["price"]
    
# Determine discount
    if category == "Electronics":
        if price >= 1000:
            discount_percent = 20
        elif price >= 500:
            discount_percent = 15
        else:
            discount_percent = 10
    elif category == "Clothing":
        if price >= 100:
            discount_percent = 25
        else:
            discount_percent = 15
    elif category == "Books":
        discount_percent = 10
    else:
        discount_percent = 0

#Calculate discounts and final price
    discount_amount = price * (discount_percent / 100)
    final_price = price - discount_amount


# 3. Output

    print(f"Product: {name}")
    print(f"Category: {category}")
    print(f"Original Price: ${price:.2f}")
    print(f"Discount: {discount_percent}%")
    print(f"Final Price: ${final_price:.2f}\n")

# Update totals
    total_original += price
    total_discount_amount += discount_amount
    total_final += final_price
    product_count += 1

print("\n=== SUMMARY ===")
print(f"Total Products: {product_count}")
print(f"Total Original Price: ${total_original:.2f}")
print(f"Total Discount: ${total_discount_amount:.2f}")
print(f"Total Final Price: ${total_final:.2f}")