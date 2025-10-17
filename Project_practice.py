# 1. Input
#Data Provided: The first step is to copy and paste the provided product.
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
# Second step put the tracking variables
total_original = 0
total_discount_amount = 0
total_final = 0
product_count = 0

#For level 1 bonus
max_discount_amount = 0 # For Level 1 Bonus
max_discount_name = "" # For Level 1 Bonus
total_discount_percent = 0 # For calculating average discount percent

# Level 2: Count products by category
category_counts = {}

# Level 2: Most and least expensive product after discount
max_final_price = -1
min_final_price = 999999
max_final_name = ""
min_final_name = ""


print("PRODUCT DISCOUNT CALCULATOR") #Name for the program

# 2. Process
#The third step is to loop through each product
#
for product in products:    #Product is each item in the products list "Products"
    name = product["name"]
    category = product["category"]
    price = product["price"]

    # Count products by category #Level 2
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1
    
# Determine discount  #Also one important step is the indentation here,if one is wrong python will give an error
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

        # Level 3: Clearance tag
    if discount_percent >= 20:
        print("Tag: Clearance")

    print()  # Blank line for better readability


# Update totals
    total_original += price
    total_discount_amount += discount_amount
    total_final += final_price
    product_count += 1
    total_discount_percent += discount_percent # For calculating average discount percent

#Level 1 Bonus amount
#Find and display the product with the highest discount amount
# Check if this is the highest discount so far
    if discount_amount > max_discount_amount:  #If the discount amount is greater than the current max
        max_discount_amount = discount_amount
        max_discount_name = name


    # Level 2: Most and least expensive after discount
    if final_price > max_final_price:
        max_final_price = final_price
        max_final_name = name
    if final_price < min_final_price:
        min_final_price = final_price
        min_final_name = name


# Display summary with the update totals

print("\n=== SUMMARY ===")
print(f"Total Products: {product_count}")
print(f"Total Original Price: ${total_original:.2f}")
print(f"Total Discount: ${total_discount_amount:.2f}")
print(f"Total Final Price: ${total_final:.2f}")
print("Tag: Clearance")

#Bonus Level 1 print
print(f"Highest Discount Product: {max_discount_name} with a discount of ${max_discount_amount:.2f}")
print(f"Average Discount Percentage: {total_discount_percent / product_count:.2f}%")


# Level 2: Category counts
print("Product count by category:")
for cat in category_counts:
    print(f"  {cat}: {category_counts[cat]}")

print(f"Most expensive product after discount: {max_final_name} (${max_final_price:.2f})")
print(f"Cheapest product after discount: {min_final_name} (${min_final_price:.2f})")

# Level 3: Total savings
print(f"Total Savings for Customer: ${total_discount_amount:.2f}")
