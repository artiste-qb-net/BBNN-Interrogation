from pprint import pprint

a = []
for i in range(3):
    a.append(set())
a[2].add(2)
a[1].add(67)
a[0].add(2)
a[2].add(34)
a[1].add(67)
a[1].add(67)
a[0].add(6867)
a[1].add(2)

pprint(a)