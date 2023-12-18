# matlab to python

% -> #

; -> 

from casadi import *
->
from casadi import *


print\('(.*)'\)
print('$1')

print\(\['(.*)'\]\)
print(f'$1')

keyboard
import pdb; pdb.set_trace()


range((([^))]*))
range($1)

\s*end
->
nothing


if (.*)
if $1:

else
else:

num2str
str

for ([a-z_]*) =
for $1 in

length\(
len(