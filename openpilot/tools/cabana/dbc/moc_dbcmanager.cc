/****************************************************************************
** Meta object code from reading C++ file 'dbcmanager.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "dbcmanager.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'dbcmanager.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_DBCManager_t {
    QByteArrayData data[13];
    char stringdata0[140];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_DBCManager_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_DBCManager_t qt_meta_stringdata_DBCManager = {
    {
QT_MOC_LITERAL(0, 0, 10), // "DBCManager"
QT_MOC_LITERAL(1, 11, 11), // "signalAdded"
QT_MOC_LITERAL(2, 23, 0), // ""
QT_MOC_LITERAL(3, 24, 9), // "MessageId"
QT_MOC_LITERAL(4, 34, 2), // "id"
QT_MOC_LITERAL(5, 37, 21), // "const cabana::Signal*"
QT_MOC_LITERAL(6, 59, 3), // "sig"
QT_MOC_LITERAL(7, 63, 13), // "signalRemoved"
QT_MOC_LITERAL(8, 77, 13), // "signalUpdated"
QT_MOC_LITERAL(9, 91, 10), // "msgUpdated"
QT_MOC_LITERAL(10, 102, 10), // "msgRemoved"
QT_MOC_LITERAL(11, 113, 14), // "DBCFileChanged"
QT_MOC_LITERAL(12, 128, 11) // "maskUpdated"

    },
    "DBCManager\0signalAdded\0\0MessageId\0id\0"
    "const cabana::Signal*\0sig\0signalRemoved\0"
    "signalUpdated\0msgUpdated\0msgRemoved\0"
    "DBCFileChanged\0maskUpdated"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_DBCManager[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       7,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   49,    2, 0x06 /* Public */,
       7,    1,   54,    2, 0x06 /* Public */,
       8,    1,   57,    2, 0x06 /* Public */,
       9,    1,   60,    2, 0x06 /* Public */,
      10,    1,   63,    2, 0x06 /* Public */,
      11,    0,   66,    2, 0x06 /* Public */,
      12,    0,   67,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3, 0x80000000 | 5,    4,    6,
    QMetaType::Void, 0x80000000 | 5,    6,
    QMetaType::Void, 0x80000000 | 5,    6,
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void DBCManager::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<DBCManager *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->signalAdded((*reinterpret_cast< MessageId(*)>(_a[1])),(*reinterpret_cast< const cabana::Signal*(*)>(_a[2]))); break;
        case 1: _t->signalRemoved((*reinterpret_cast< const cabana::Signal*(*)>(_a[1]))); break;
        case 2: _t->signalUpdated((*reinterpret_cast< const cabana::Signal*(*)>(_a[1]))); break;
        case 3: _t->msgUpdated((*reinterpret_cast< MessageId(*)>(_a[1]))); break;
        case 4: _t->msgRemoved((*reinterpret_cast< MessageId(*)>(_a[1]))); break;
        case 5: _t->DBCFileChanged(); break;
        case 6: _t->maskUpdated(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (DBCManager::*)(MessageId , const cabana::Signal * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DBCManager::signalAdded)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (DBCManager::*)(const cabana::Signal * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DBCManager::signalRemoved)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (DBCManager::*)(const cabana::Signal * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DBCManager::signalUpdated)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (DBCManager::*)(MessageId );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DBCManager::msgUpdated)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (DBCManager::*)(MessageId );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DBCManager::msgRemoved)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (DBCManager::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DBCManager::DBCFileChanged)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (DBCManager::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DBCManager::maskUpdated)) {
                *result = 6;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject DBCManager::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_DBCManager.data,
    qt_meta_data_DBCManager,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *DBCManager::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DBCManager::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_DBCManager.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int DBCManager::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void DBCManager::signalAdded(MessageId _t1, const cabana::Signal * _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void DBCManager::signalRemoved(const cabana::Signal * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void DBCManager::signalUpdated(const cabana::Signal * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void DBCManager::msgUpdated(MessageId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void DBCManager::msgRemoved(MessageId _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void DBCManager::DBCFileChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void DBCManager::maskUpdated()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
