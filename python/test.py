class A:
    def __init__(self):
        print("__init__ a")

    def __call__(self):
        print("__call__ a")



a = A()

# b = a()