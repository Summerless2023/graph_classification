# Definition for singly-linked list.
class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None


def mergeTwoLists(l1, l2):
    ans = None
    while l1 is not None and l1.next is not None and l2 is not None and l2.next is not None:
        if l1.val <= l2.val:
            if ans is None:
                ans = ListNode(l1.val)
            else:
                ans.next = ListNode(l1.val)
                ans = ans.next
            l1 = l1.next
        elif l2.val < l1.val:
            if ans is None:
                ans = ListNode(l2.val)
            else:
                ans.next = ListNode(l2.val)
                ans = ans.next
            l2 = l2.next
    while l1 is not None and l1.next is not None:
        ans.next = ListNode(l1.val)
        ans = ans.next
        ans.next is None
        l1 = l1.next
    while l2 is not None and l2.next is not None:
        ans.next = ListNode(l2.val)
        ans = ans.next
        ans.next is None
        l2 = l2.next
    return ans

a = ListNode(1)
b = ListNode(2)
c = ListNode(3)

a.next = b
b.next = c
d = ListNode(1)
e = ListNode(2)
f = ListNode(4)
d.next = e
e.next = f

mergeTwoLists(a,d)