var pizza1 = context.forms.Order.pizzasize;
var discount = context.entities.price2;

if (pizza1 == "Large - 549/-"){
    discount = 549 * 0.5
}
else if(pizza1 == "Extra Large - 799/-"){
    discount = 799 * 0.5
}
else if(pizza1 == "Medium - 399/-"){
    discount = 399
}
else{
    discount = 259
}

context.entities.price2 = discount;