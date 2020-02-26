class GrandFather:
    def __init__(self):
        self.grand_parent_value = "grand_father_value"

    def get_grand_parent_value(self):
        return self.grand_parent_value


class GrandMother:
    def __init__(self):
        self.grand_parent_value = "grand_mother_value"

    def get_grand_parent_value(self):
        return self.grand_parent_value


def create_parent(base):
    class Parent(base):
        def __init__(self):
            super(Parent, self).__init__()
            self.parent_value = "parent_value"

        def get_parent_value(self):
            return self.parent_value
    return Parent


def create_child(base):
    class Child(create_parent(base)):
        def __init__(self):
            super(Child, self).__init__()
            self.child_value = "child_value"

        def get_child_value(self):
            return self.child_value
    return Child

child = create_child(GrandFather)()
print(child.get_child_value())
print(child.get_parent_value())
print(child.get_grand_parent_value())

child = create_child(GrandMother)()
print(child.get_child_value())
print(child.get_parent_value())
print(child.get_grand_parent_value())