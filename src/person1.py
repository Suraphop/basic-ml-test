# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#create classes

class Myclass:
    x = 5


# %%
#create object

p1 = Myclass()
print(p1.x)


# %%
#init fuction

class Person:
    def __init__(self, names ,surname, ages):
        self.name = names
        #self.name = "this is "+names
        self.surname = surname
        self.age = ages+10
        
    def myfunc(self):
        print("hello my name is "+ self.name +' '+ self.surname)

p2 = Person('Suraphop','Bunsawat',29)
print(p2)
print(p2.name)
print(p2.age)

p2.myfunc()


