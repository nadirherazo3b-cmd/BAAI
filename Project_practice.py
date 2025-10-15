# List of products with name, price, and category
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

def calculate_discount(product):
    category = product["category"]
    price = product["price"]
    if category == "Electronics":
        if price >= 1000:
            discount = 0.20
        elif price >= 500:
            discount = 0.15
        else:
            discount = 0.10
    elif category == "Clothing":
        if price >= 100:
            discount = 0.25
        else:
            discount = 0.15
    elif category == "Books":
        discount = 0.10
    else:
        discount = 0
    return discount

# Process products
summary = []
total_original = 0
total_discount = 0
total_final = 0

for product in products:
    discount = calculate_discount(product)
    discount_amount = product["price"] * discount
    final_price = product["price"] - discount_amount
    summary.append({
        "name": product["name"],
        "category": product["category"],
        "original_price": product["price"],
        "discount_applied": f"{discount * 100:.0f}%",
        "final_price": final_price
    })
    total_original += product["price"]
    total_discount += discount_amount
    total_final += final_price

# Display summary
print("Product Summary:")
print("Name       Category      Price    Discount    Final Price")
for s in summary:
    print(f'{s["name"]:10} {s["category"]:12} ${s["original_price"]:7.2f} {s["discount_applied"]:10} ${s["final_price"]:12.2f}')

print("\nTOTALS:")
print(f"Total products: {len(products)}")
print(f"Total original price: ${total_original:.2f}")
print(f"Total discount: ${total_discount:.2f}")
print(f"Total final price: ${total_final:.2f}")