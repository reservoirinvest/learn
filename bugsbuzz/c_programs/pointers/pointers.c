#include <stdio.h>
#include <string.h>
int main()
{
   char q[10] = "I am God!@";
   /* address of q is assigned to ptr */

   /* display q's value using ptr variable */
   printf("*(q+1) is %p\n", (q+1));
   printf("&q[1] is %p\n", &q[1]);
   return 0;
}
