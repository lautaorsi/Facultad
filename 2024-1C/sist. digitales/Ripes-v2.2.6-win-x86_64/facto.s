.data:
    n: .word 0x08
    
.text:
    la a0 n
    lw a0 0(a0) #cargamos el nro a a0
    addi a1, zero, 1 #inicializamos un valor de retorno en a1 con valor 1
    facto:
        beq a0, zero, end
        mul a1 a1 a0
        addi a0 a0 -1
        j facto
    end:
        addi x0, t0 1