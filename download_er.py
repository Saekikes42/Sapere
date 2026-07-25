import requests

mermaid_code = """graph TD
    U[Пользователь]
    M[Учебный материал]
    S[Пакет сканирования]

    rel1{Инициирует}
    rel2{Включает}
    rel3{Анализирует}

    U ---|1| rel1
    rel1 ---|M| S

    S ---|1| rel2
    rel2 ---|M| M

    U ---|1| rel3
    rel3 ---|M| M

    U_id([ID пользователя])
    U_name([Имя / Никнейм])
    U --- U_id
    U --- U_name

    M_id([ID файла])
    M_name([Название])
    M_subj([Предмет])
    M_diff([Сложность])
    M --- M_id
    M --- M_name
    M --- M_subj
    M --- M_diff

    S_id([ID пакета])
    S_date([Дата])
    S_count([Количество])
    S --- S_id
    S --- S_date
    S --- S_count

    classDef entity fill:#bbdefb,stroke:#333,stroke-width:2px;
    classDef relation fill:#c8e6c9,stroke:#333,stroke-width:2px;
    classDef attribute fill:#fff9c4,stroke:#333,stroke-width:1px;

    class U,M,S entity;
    class rel1,rel2,rel3 relation;
    class U_id,U_name,M_id,M_name,M_subj,M_diff,S_id,S_date,S_count attribute;
"""

url = 'https://kroki.io/mermaid/png'
payload = {'diagram_source': mermaid_code}
response = requests.post(url, json=payload)

if response.status_code == 200:
    with open('c:/sapere-2/Пояснительная_записка_черновики/er_diagram.png', 'wb') as f:
        f.write(response.content)
    print("Success")
else:
    print(f"Failed: {response.status_code}")
