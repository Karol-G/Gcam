# class GrandFather:
#     def __init__(self):
#         self.grand_parent_value = "grand_father_value"
#
#     def get_grand_parent_value(self):
#         return self.grand_parent_value
#
#
# class GrandMother:
#     def __init__(self):
#         self.grand_parent_value = "grand_mother_value"
#
#     def get_grand_parent_value(self):
#         return self.grand_parent_value
#
#
# def create_parent(base):
#     class Parent(base):
#         def __init__(self):
#             super(Parent, self).__init__()
#             self.parent_value = "parent_value"
#
#         def get_parent_value(self):
#             return self.parent_value
#     return Parent
#
#
# def create_child(base):
#     class Child(create_parent(base)):
#         def __init__(self):
#             super(Child, self).__init__()
#             self.child_value = "child_value"
#
#         def get_child_value(self):
#             return self.child_value
#     return Child
#
# child = create_child(GrandFather)()
# print(child.get_child_value())
# print(child.get_parent_value())
# print(child.get_grand_parent_value())
#
# child = create_child(GrandMother)()
# print(child.get_child_value())
# print(child.get_parent_value())
# print(child.get_grand_parent_value())

class Test3(object):
    def do_it2(self):
        print("I have done it2!")

class Test2(Test3):
    def do_it(self):
        print("I have done it!")

def create_test(base):
    class Test(base):
        def __init__(self, test2):
            super(Test, self).__init__()
            self.tmp = 5
            self.test2 = test2
            self.test2.do_it()

        def __getattr__(self, method):
            print("TMP-------------------------- ABSTRACT METHOD GCAM HOOK (" + method + ") --------------------------")

            def abstract_method(*args):
                print("-------------------------- ABSTRACT METHOD GCAM HOOK (" + method + ") --------------------------")
                #return self.model.method(args)

            return abstract_method

        def forward(self, batch):
            print(batch)
    return Test

test2 = Test2()
base = type(test2).__bases__[0]
#test = Test(Test2())
test = create_test(base)(test2)
print(test.tmp)
test.forward("haha")
test.gulag()
test.do_it2()