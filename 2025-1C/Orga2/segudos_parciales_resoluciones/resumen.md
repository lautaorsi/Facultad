/* MMU */
VIRT_PAGE_OFFSET(X) devuelve el offset dentro de la página, donde X es una dirección virtual
VIRT_PAGE_TABLE(X)  devuelve la page table INDEX correspondiente, donde X es una dirección virtual
VIRT_PAGE_DIR(X)    devuelve el page directory INDEX, donde X es una dirección virtual
CR3_TO_PAGE_DIR(X)  devuelve el page directory, donde X es el contenido del registro CR3
MMU_ENTRY_PADDR(X)  devuelve la dirección física de la base de un page frame o de un page table, donde X es el campo de 20 bits en una PTE o PDE


SCHEDULER

Estructura usada por el scheduler para guardar la información pertinente de
cada tarea.

typedef struct {
  int16_t selector;
  task_state_t state;
} sched_entry_t;

static sched_entry_t sched_tasks[MAX_TASKS] = {0};

uint8_t current_task -> id de tarea ejecutandose actualmente

