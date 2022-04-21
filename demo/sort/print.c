#include <stdio.h>

int main(int argc, char **argv)
{
    FILE *fp;
    int a = 123456789;
    fp = fopen("test.a","w");
    fprintf(fp, "%d", a);
    fclose(fp);
    fp = fopen("test.b","wb");
    fwrite(&a, 1, sizeof(a), fp);
    fclose(fp);
    return 0;
}