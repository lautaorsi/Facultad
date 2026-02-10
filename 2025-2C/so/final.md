# Preguntas de Final (Oral)

## Procesos

### ¿Qué es una system call? ¿Para qué se usan? ¿Cómo funcionan? Explicar en detalle el funcionamiento una system call en particular.

### ¿Las system calls son universales a todos los sistemas operativos o son dependientes de cada sistema?

### ¿Para qué sirve la system call fork? ¿Qué debilidades tiene? Comparar con vfork y la creación de threads.

### Diferencias entre system calls para crear procesos entre Linux y Windows.

### ¿Cómo funcionan los estados de un proceso? Ready, bloqueado, running. Explicar las transiciones de cada estado a cada estado (en particular, de waiting a ready).

### ¿Qué estructura se debe mantener en memoria para poder tener procesos? Hablar de la tabla de procesos.

### ¿Qué es un proceso, un thread y en qué se diferencian?

Un proceso es una instancia de ejecución de un programa, tiene su propia sección de código, datos, stack, heap, registros e instruction pointer. <br> Por otro lado, los threads son las menores unidades de ejecución de la CPU y "fragmenta" al proceso para poder realizar distintas tareas en "simultaneo" (no realmente en simultaneo, pero hace que puedas distribuir responsabilidades y que no se cuelge el programa). Cada thread tiene su propio stack, heap, registros e instruction pointer pero comparten datos y código. Los threads son menos pesados que los procesos, pero al compartir datos es importante usarlo con cuidado. <br> Un ejemplo de threads: si tenemos un programa que realiza una operación de I/O otro thread del mismo proceso puede seguir realizando operaciones para que el usuario no note un "cuelgue".

### ¿Qué debería agregar a la PCB para manejar los threads?

Una **TCB** o *Thread Control Block* para cada thread, basicamente una mini PCB por thread que retenga la informacion unica de cada uno (vease la respuesta al item anterior)

### ¿Qué pasaría si los threads compartieran el stack?

Se rompe todo, el stack es relativo y secuencial, si thread A guarda algo en el stack, despues thread B se ejecuta pusheando 5 datos al stack y luego A quiere acceder al dato, como sabe que se pushearon 5 datos? <br> Tendriamos que tener un registro de todos los pusheos o de los offsets relativos al stack pointer de cada dato guardado por los threads -muy engorroso y poco efectivo-.

### Qué tendría que ver en un sistema para que piense que va a andar mejor agregando:

- más procesadores

Que el SO tenga demasiados procesos en estado *Ready* sin ser ejecutados.

- más memoria.

Que el SO este haciendo *thrashing*, donde basicamente tenemos tan pocos frames que estamos constantemente sacando y poniendo paginas de la memoria por frames insuficientes.

### Hablar de strace y ptrace.

No vimos ptrace, strace the muestra las syscalls de los procesos

## Scheduling

### Describir los objetivos que pueden tener las políticas de scheduling (fairness, carga del sistema, etc.).

La idea de las politicas de scheduling es poder ejecutar todos los procesos de una forma "justa", la cuestión está en como interpretamos esta justicia entre procesos. <br>
En lineas generales es importante dar un poco mas de prioridad a procesos interactivos ya que estos son los mas notorios para el usuario, pero tampoco queremos causar inanición de los procesos no interactivos o de computos pesados.

### ¿Qué objetivo prioriza SJF y por qué no se usa en la práctica?

SJF o *Shortest Job First* prioriza los procesos que vayan a ejecutarse por menos tiempo, para esto necesitamos en general un sistema de dispatch en forma de batch, el problema es que necesitamos predecir la duración y esto puede fallar.

### ¿Cómo funciona el scheduling con múltiples colas?

Se arman X colas con 0 la de mayor prioridad y X-1 la de menor, esto se combina con un quantum inverso a la prioridad (Mayor prioridad -> Menor quantum) y algun *boost* a procesos interactivos. Además a medida que cada proceso es termina su correspondiente quantum se degrada a una cola anterior (o puede tener varias "chances", dependiendo de la configuración), al mismo tiempo cada cola puede tener un esquema particular de scheduling como RR.

### ¿Hay algún problema con que las prioridades fueran fijas?

Inanición de los procesos de menor prioridad, si tenemos uno en la cola 1 y otro en 0 que no se mueve de ahí el proceso de menor prioridad va a quedar colgado hasta que el mayor termine.  

### Hablar sobre la afinidad de un procesador. ¿Qué información extra tenemos que tener en la PCB para mantener afinidad en un sistema multicore?

No se que es afinidad

### Explicar el problema de inversión de prioridades.

Propongo la situación A B y C con prioridades A > B > C, A y C comparten un recurso crítico y M no. <br>
C toma control del recurso y se ejecuta pero es interrumpido por A quien necesita usar el recurso y por lo tanto debe esperar a que C lo libere. B despierta y al ser de mayor prioridad que C pasa a ser ejecutado, como B no usa la sección crítica impide que C finalice e indirectamente impide que A sea ejecutado, causando que un proceso de mayor prioridad (B) bloquee a uno de mayor prioridad (A).


## Sincronización

### ¿Para qué necesitamos sincronización entre procesos? ¿Qué soluciones nos ofrece el HW? Explicar el caso para monoprocesador y para multiprocesador. (instrucciones atómicas y deshabilitar interrupciones)

Cuando tenemos procesos siendo ejecutados de forma simultanea y datos compartidos entre ellos es evidente que las race conditions pueden causar inconsistencias peligrosas para el funcionamiento correcto, el HW nos ofrece instrucciones atomicas que no pueden ser interrumpidas para asegurar que se termine de ejecutar la instrucción. Un ejemplo es el **TAS** o *Test And Set* que verifica una variable y la cambia a 1 si estaba en 0. <br>

### ¿Cómo nos afecta si el scheduler es preemptive o non-preemptive en la implementación de un semáforo?

Si el preemptiveness no se ve afectado por la deshabilitacion de las interrupciones entonces habría que tener cuidado con el contexto en el cual la CPU va a desalojar a un proceso en ejecución. Si al deshabilitar las interrupciones no puede desalojar entonces lo unico que cambiaria es que al habilitar un semaforo el proceso de mayor prioridad que estaba esperando recibiria ejecución.

### Evaluar si están bien o mal utilizadas en los siguientes ejemplos las primitivas de sincronización:

- #### Usar TASLock (spinlock) para acceder a disco.

Es un poco raro el planteo, spinlock es la idea de hacer busy waiting sobre un mutex hasta que se libere. <br>
Dicho eso, si la idea es obtener el disco todo va a depender de quien obtenga CPU luego de que el disco sea liberado, si justo aparece uno de mayor prioridad el spinlock es inútil ya que el nuevo proceso va a ejecutarse antes y tomarlo. <br>
Por otro lado, usar un ``.wait()`` tampoco te garantiza ser el que obtiene el acceso al disco, todo va a depender de como sea manejada la queue de procesos bloqueados, si es un esquema FIFO y nadie mas lo pidió entonces ganas, pero es muy situacional.

- #### Usar semáforos para incrementar un contador.

Si, esta bien. Probablemente sería mas óptimo usar la suma atómica ``ADDR`` pero no va a pasar nada grave por usar un mutex.

- #### Usar un contador atómico para un recurso que tienen que poder acceder 3 procesos a la vez.

Esta bien, es basicamente la idea de usar un semaforo inicializado en 3

- #### Usar spinlock para un recurso que tienen que poder acceder 3 procesos a la vez.

Esta mal, un spinlock solo habilita a un proceso a la vez ya que depende de un mutex

### Diferencia entre spin lock y semáforos (hablar de TTAS). ¿En qué contexto preferimos uno sobre el otro y por qué?.

Si podemos asegurar que va a liberarse dentro de poco el lock entonces es mas útil spinlock, en particular si sabemos que va a tardar menos en liberarse que lo que tardaría hacer un context switch, además de lo dicho en la pregunta anterior. <br>
**Esto solo es util en un esquema multiprocesador, no sirve de nada usar spinlock si bloqueamos completamente al proceso que esta usando el lock**

### ¿Cómo implementamos una sección crítica con spin locks?

Los procesos hacen TAS hasta que se libera el dato, una vez liberado tomamos control bloqueandolo y hacemos lo que haya que hacer, despues lo liberamos

    while(TAS(mutex))
    //seccion critica
    mutex.release()

### Explicar el problema clásico de lectores y escritores. Explicar cómo podemos evitar inanición del escritor.

Llamemoslos L y E respectivamente, podemos plantear dos problemas: <br> <br>

*Lost Wake-Up call*: si E hace una escritura pero una race condition (L chequea buffer antes de que E pudiera registrar el valor en memoria -se llego a hacer el add pero no el push-) causa que L pierda la Wake-Up call, con un esquema de escritura-lectura bloqueante (E no puede escrbir hasta que L lea) generamos que tanto L como E queden colgados para siempre.
<br><br>
*Writer starvation*: si tenemos multiples lectores y un mutex para asegurar que no pueda escribirse y leerse en simultaneo el buffer (pero si leerse en simultaneo) puede ocurrir que L1 entra a leer, E quiere escrbir pero como esta L1 adentro se queda esperando, luego L2 aparece y entra a leer (pues la lectura no causa race conditions por si sola), luego L1 sale pero L2 sigue adentro y por lo tanto E sigue bloqueado, aparece L3 y entra a leer .... E queda eternamente bloqueado y cualquier Ln que ingrese a leer va a encontrar valores viejos.

## Memoria

### Se nos muestra un árbol de procesos donde cada proceso tiene una serie de page frames asignados. Explicar las siguientes situaciones:

- ### ¿Por qué todos los procesos de un sistema compartirían una página? (páginas del kernel o bibliotecas compartidas)

Es evidente que tener una declaración de funciones estándar como `printf` por cada proceso es altamente ineficiente, entonces es mejor tener una serie de paginas compartidas sin permisos de escritura que permitan a todos los procesos utilizarlas on-demand. <br> A nivel kernel ocurre algo parecido si tenemos syscalls 

- ### ¿Por qué dos procesos específicos podrían compartir una página? (hablar de fork y copy-on-write)

Recordemos que los procesos siempre son **forks** de otros (Teniendo como raiz al kernel encargado de bootear), luego cada proceso puede hacer execv para cambiar su ejecución o mantener el mismo código, pero como tener duplicadas todas las páginas cuando simplemente duplicamos un proceso para que haga lo mismo que le original es altamente ineficiente se propone hacer **copy-on-write** para solo tener paginas distintas cuando realmente haya una diferencia.

### ¿Para qué sirve la paginación de la memoria? ¿Qué ventajas tiene sobre utilizar direcciones físicas? Hablar sobre el tamaño de las páginas y cómo se relaciona con el tamaño de los bloques en disco. (hablar de fragmentación interna y fragmentación externa)

En general, los sistemas modernos usan una relación 1:1 entre tamaño de pagina y tamaño de bloque. Las ventajas que tiene usar paginación es la de usar espacio no-contiguo (fragmentación externa), seguridad entre procesos y na mayor claridad a nivel codigo por las direcciones virtuales. Utilizar paginación tiene la desventaja de desperdiciar el espacio no utilizado dentro de las paginas mismas (si tenemos bloques de X Kb y usamos X+1 Kb vamos a desperdiciar X-1 Kb)

### ¿Qué es un page fault y cómo se resuelve?

Es cuando un proceso que esta siendo ejecutado necesita informacion que no esta contenida en los frames de la memoria RAM, se resuelve reemplazando algun frame (dependerá de la politica de desalojo) por el dato requerido.

### ¿Por qué puede pasar que tengamos muchos procesos en waiting, y cómo podría tratar de arreglarlo si no pudiese agregar memoria?

Podemos tener muchos procesos en waiting cuando ocurre una gran cantidad de operaciones E/S, se podría arreglar acelerando el tiempo de estas operaciones, en el contexto de E/S a disco usar un buen esquema de distribución es fundamental así como tambien lo es la politica de scheduling de E/S del propio disco.

## E/S

### Hablar de RAID (para qué sirve). Explicar la diferencia entre RAID 4 y RAID 5. ¿Cuál es mejor y por qué?

RAID es la idea de usar multiples discos con redundancia para asegurar la integridad de la información, si se pierde un disco y tenemos backups podemos estar un poco mas tranquilos que si perdemos todo. <br>
RAID 4 y RAID 5 son muy similares a nivel cantidad de redundancia pero fundamentalmente distintos en si organización, basicamente la idea es tener un disco de paridad que hace XOR's sobre los bloques que permite recuperar cualquier disco que se pierda. <br> La cuestión está en como se distribuye este disco de paridad, en **RAID 4** es un unico disco el encargado de contener toda la paridad y esto causa que si modificamos **cualquier** dato entonces el disco de paridad tambien tiene que modificarse (esto hace que el disco de paridad este laburando constantemente), mientras que **RAID 5** distribuye la paridad entre los discos, esto asegura un trabajo un poco mas distribuido. *(RAID 5 es mejor)*

### Explicar los distintos algoritmos básicos de scheduling de disco. Explicar cómo se comportan en HDD y en SDD.

Los algoritmos que tenemos son:<br>
- *FCFS*: el primero que llega es el primero que se hace, es poco eficiente porque podemos estar yendo y viniendo indiscriminadamente de manera muy naive.
- *Elevator*: la cabeza se mueve como ascensor que va a ir subiendo y bajando "barriendo" el disco, si llega un pedido y justo la cabeza acaba de pasarse del lugar esta forzado a esperar a que de toda la vuelta
- *SSTF*: la cabeza se mueve a la posicion que mas cerca tiene, esto puede causar inanicion para algun dato que este muy lejos 

### ¿Qué son los drivers y qué permiten?

Son traductores entre un dispositivo especfico y el SO, corren con privilegios root entonces es importantisimo que esten bien diseñados, los tipos de comunicacion que pueden tener son:
- Polling: un while para chequear si el dispositivo se comunicó. CONSUME CPU
- Interrupciones: levanta interrupciones si el dispositivo se comunicó. CONTEXT SWITCH IMPREDECIBLE
- DMA: el driver va a modificar directamente el buffer. REQUIRE HW 

### Explicar diferencias entre un disco magnético y un SSD. ¿Qué problemas tiene un SSD? Hablar de write amplification y borrado.

El disco magnetico es literalmente un disco que gira a altas velocidades, SSD es electronico y no tiene rotaciones

### ¿Cómo hace un disco para saber si un bloque está libre / si puede ser borrado? Explicar cómo podemos indicarle a un SSD que un bloque está libre (y que puede borrado). (comando TRIM)

### Explicar cómo se puede hacer una recuperación de datos, después de haber borrado un archivo.

Tenemos que conseguir las direcciones fisicas que le habia asignado el SO para mandarselas al disco y que este pueda traducirlas a los bloques.

## FS

### ¿Qué es un file descriptor? Nombrar 3 system calls que los afecten.

Un file descriptor como tal es un indice de una entry en la file descriptor table, pero basicamente referencia archivos abiertos por procesos especificos. <br>
Dentro de las syscalls que los afecten tenemos **open**, **close** y **dup2** que modifica la entry en la tabla (es el tipico usado para reemplazar -por ejemplo- STDOUT de un proceso con el WRITE de un pipe)

### ¿Cuándo se revisan los permisos de acceso sobre un archivo? Explicar por qué el file descriptor se crea cuando hacemos un open y no se vuelven a revisar los permisos.

### ¿Qué es un FS y para qué sirve?

Un filesystem es una forma de organizar y dar metadatos sobre cada archivo contenido en discos montados en la maquina.

### ¿Cuándo es adecuado reservar espacio en disco de manera secuencial? ¿Qué beneficios nos trae? (CD-ROM, ISO-9660)

### ¿Cuál FS nos conviene utilizar para un sistema embebido: FAT o inodos?

### ¿Cuál FS nos conviene utilizar para implementar UNDO (deshacer la última escritura a disco)? ¿Cómo se implementaría en FAT y en inodos?

### ¿Cuál FS nos conviene utilizar para implementar un backup total? ¿Cómo se implementaría en FAT y en inodos?

### ¿Cuál FS nos conviene utilizar para implementar un backup incremental? ¿Cómo se implementaría en FAT y en inodos?

### ¿Cuál FS nos conviene utilizar para implementar snapshot? (Diferenciar el caso en que queramos tomar una snapshot mientras el sistema está corriendo) ¿Cómo se implementaría en FAT y en inodos?

### Explicar las diferencias entre FAT e inodos. Ventajas y desventajas de cada uno.

FAT es una tabla donde cada indice de bloque apunta al siguiente bloque del archivo (o EOF/0 si es el ultimo o esta vacio, respectivamente), el problema es la contención de la informacion (FAT contiene TODO y si se rompe estas en problemas) y que la FAT siempre debe estar cargada en memoria, entonces con discos muy grandes llenas la RAM solo con metadatos. <br>
Inodos permite una distribucion de los bloques de cada archivo ya que no esta centralizado, pero es un sistema mas sofisticado que usa bastantes metadatos generales, lo bueno es que no depende del tamaño del disco.

### ¿FAT implementa algún tipo de seguridad?

No

### Explicar qué es journaling.

Es la idea de mantener un registro sobre los cambios a realizar en disco, esto es mas que nada para asegurar que si se apaga la maquina de forma repentina podemos hacer un chequeo mas educado de los cambios que no pudieron hacerse sin tener que verificar todo el sistema

### Describir ext2.

La idea es tener grupos de bloques que contienen inodos y un superbloque con toda la informacion critica del sistema. 

### ¿Qué mantiene un inodo? ¿Cómo es la estructura de directorios?

Contiene algunos metadatos sobre el archivo y sus datos en direcciones e indirecciones. <br>
Los directorios contienen estructuras dirEntry que contienen metadatos como el nombre del archivo y el inodo que lo representa

### ¿Para qué sirven los block groups y los clusters? Motivación para agrupar páginas en bloques, y bloques en grupos de bloques.



### ¿Cuáles son las estructuras más importantes de ext2? Explicar cada una (en particular, hablar del superbloque).

### Explicar cómo se manejan los bloques libres del disco.

### ¿Qué pasa cuando se inicia el sistema luego de haber sido apagado de forma abrupta? Explicar cómo hace el sistema para darse cuenta de si hubo algún error (cuando no hace journaling) y cómo lo arregla. (inconsistencias entre contadores y bitmaps, entre entradas de inodos y bitmaps, entre entradas de directorios y bitmaps)

### Explicar las diferencias (ventajas y desventajas) entre soft links y hard links. ¿Por qué no podemos usar un hard link para referenciar inodos de otro FS, incluso si está basado en inodos?

### Explicar cómo se crean y borran archivos con las estructuras del FS (incluido cómo se modifica el block group). Explicar el caso de borrado en hard links.

### Explicar qué ocurre cuando se borra un archivo en ext3 (y diferencias con ext2).

## Distribuidos

### ¿Cómo podemos mantener una base de datos distribuida sincronizada?

### ¿Qué es un sistema distribuido? ¿Qué significa que un sistema sea completamente distribuido? ¿Qué beneficios ofrecen? ¿Qué problemas nos puede traer?

### Explicar los distintos algoritmos de commits que se utilizan para actualizar una copia del archivo que está en varios nodos (2PC y 3PC). ¿Cuál es la diferencia entre 2PC y 3PC? Explicar la diferencia entre weak/strong termination.

### ¿Qué hace un nodo si se cae después de que todos digan sí al coordinador en 2PC?

### Explicar el problema bizantino.

### Explicar token ring ¿Qué problemas tiene? ¿Qué se hace en caso de que se pierda el token? ¿Cómo podemos mejorarlo?

### Explicar la diferencia entre grid y cluster.

### File Systems Distribuidos

### ¿Qué es un file system distribuido? Explicar la interfaz VFS.

### Hablar de las limitaciones en DFS.

### ¿Cómo podría hacer para poder tener 2 discos distribuidos, y que los dos contengan la misma información? Y en caso de que se caiga la conexión, ¿cómo hacemos?

### ¿NFS es un file system completamente distribuido?

### Proponer una manera para mantener sincronizados N servidores NFS. Explicar cómo saben los nodos a qué servidor pedirle los datos.

### Si un nodo se cae, ¿cómo hacemos para que se entere después de las transacciones que no tiene?

## Seguridad

### ¿Cuáles son las tres categorías principales de permiso frente a un archivo? (owner, group, universo, ACL, capability)

### Explicar cómo son las ACLs en UNIX.

### Explicar SETUID. Explicar otro método (distinto a setuid y setguid) para ejecutar con permisos de otro usuario.

### Explicar cómo funcionan los canarios.

### ¿Para qué sirve el bit NX (No eXecute) de la tabla de páginas?

### Explicar buffer overflow y mecanismos para prevenirlo. (canario, páginas no ejecutables, randomización de direcciones) ¿En qué parte de la memoria se buscan hacer los buffer overflow? (En el stack, pero también se puede hacer en el heap.

### ¿Cómo se genera el canario? ¿En tiempo de compilación o de ejecución?

### Explicar el ataque Return to libc.

### Dar un ejemplo de un error de seguridad ocasionado por race condition. (cambiar concurrentemente puntero de symbolic links)

### Explicar las diferencias entre MAC y DAC.

### ¿Qué es una firma digital? Explicar cómo es el mecanismo de autenticación.

### ¿Qué es una función de hash? ¿Cómo se usa en el log-in de un SO? ¿Qué problemas tiene usar un hash para autenticar usuarios? ¿Cómo se puede mejorar la seguridad? ¿Qué es un SALT? ¿Cómo podemos hacer que calcular la función de hash sea más costoso? ¿Qué otros usos tiene la función de hash en seguridad?
### ¿Qué es una clave pública/privada? ¿Cómo podemos distribuir una clave pública?

### ¿Por qué es más seguro utilizar un esquema de clave pública/privada para una conexión por ssh, comparado con usuario/contraseña?

### ¿Cómo podemos asegurarnos que un programa es confiable? ¿Qué pasa si nos modifican tanto el hash como el archivo? Explicar cómo podemos brindar de manera segura las actualizaciones de un software. (integridad, autenticación, canal seguro)

### Explicar las diferencias entre HTTP y HTTPS. Explicar cómo se podría realizar un ataque cuando se realiza una actualización de un programa por HTTP (suponiendo que la actualización está firmada digitalmente) pegándole a un endpoint http.

### ¿Se puede considerar Deadlock un problema de seguridad?

### ¿Qué problemas de seguridad hay asociados a los file descriptors? ¿Cómo lo resuelve SELinux?

## Virtualización

### ¿Qué es la virtualización y los contenedores? ¿Cómo se implementan?
