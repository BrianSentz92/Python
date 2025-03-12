import sqlite3


def create_database():
    conn = sqlite3.connect("expenses.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS expenses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        amount REAL NOT NULL,
                        description TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()


def add_expense(category, amount, description):
    conn = sqlite3.connect("expenses.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO expenses (category, amount, description) VALUES (?, ?, ?)",
                   (category, amount, description))
    conn.commit()
    conn.close()


def view_expenses():
    conn = sqlite3.connect("expenses.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM expenses ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    return records


def main():
    create_database()
    while True:
        print("\nExpense Tracker")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            category = input("Enter category (e.g., Food, Rent, Transport): ")
            try:
                amount = float(input("Enter amount: "))
                description = input("Enter description (optional): ")
                add_expense(category, amount, description)
                print("Expense added successfully!")
            except ValueError:
                print("Invalid amount. Please enter a number.")
        elif choice == "2":
            expenses = view_expenses()
            if expenses:
                print("\nID | Category | Amount | Description | Timestamp")
                print("------------------------------------------------")
                for exp in expenses:
                    print(f"{exp[0]} | {exp[1]} | ${exp[2]:.2f} | {exp[3]} | {exp[4]}")
            else:
                print("No expenses recorded yet.")
        elif choice == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
