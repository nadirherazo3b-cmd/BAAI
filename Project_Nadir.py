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

#Process products
summary = []
total_original = 0
total_discount_amount = 0
total_final = 0

For product in products:
    discount = calculate_discount(product)
    discount_amount = product["price"] * discount
    final_price = product["price"] - discount_amount
