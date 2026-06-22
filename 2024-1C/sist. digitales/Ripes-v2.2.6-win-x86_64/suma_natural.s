.data:
    n: .word 0x08
    
.text:
    la a0 n
    lw a0 0(a0) #cargamos el nro a a0
    add t1 zero a0
 
    suma_natural:
        beq a0 zero casi_fin
        addi sp sp -4
        sw a0 0(sp)
        addi a0 a0 -1
        j suma_natural
        casi_fin:
            lw t0 0(sp)
            beq t0 t1 fin 
            add a0 a0 t0
            addi sp sp 4
            j casi_fin
            
        fin:
            add a0 a0 t1
            
            
        
        
        
        
        