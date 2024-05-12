    bits 64

    section .data
    
    ; zadani 1
    extern data

    ; zadani 2
    extern data2
    extern result

    ; zadani 3
    extern data3

    ; zadani 4
    extern key
    extern extended

    section .text

; zadani 1
global swap_endiannes

swap_endiannes:
    ; 0 a 3
    mov ah, [data + 3]
    mov al, [data + 0]
    mov [data + 3], al
    mov [data + 0], ah
    
    ; 1 a 2
    mov ah, [data + 2]
    mov al, [data + 1]
    mov [data + 2], al
    mov [data + 1], ah
    
    ret


; zadani 2
global compose

compose:
    movzx eax, byte [data2 + 3]
    shl eax, 8
    movzx ebx, byte [data2 + 2]
    or eax, ebx
    shl eax, 8
    movzx ebx, byte [data2 + 1]
    or eax, ebx
    shl eax, 8
    movzx ebx, byte [data2]
    or eax, ebx
    mov [result], eax

    ret


; zadani 3
global replace

replace:
    mov byte [data3], "I"
    mov byte [data3 + 1], "R"
    mov byte [data3 + 2], "Z"
    mov byte [data3 + 3], "0"
    mov byte [data3 + 4], "0"
    mov byte [data3 + 5], "0"
    mov byte [data3 + 6], "6"

    ret

; zadani 4
global extend

extend:
    mov ax, [key]
    movsx rax, ax
    mov [extended], rax

    ret