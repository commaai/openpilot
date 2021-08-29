/****************************************************************************
** Meta object code from reading C++ file 'offroad_alerts.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "offroad_alerts.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'offroad_alerts.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_AbstractAlert_t {
    QByteArrayData data[3];
    char stringdata0[23];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_AbstractAlert_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_AbstractAlert_t qt_meta_stringdata_AbstractAlert = {
    {
QT_MOC_LITERAL(0, 0, 13), // "AbstractAlert"
QT_MOC_LITERAL(1, 14, 7), // "dismiss"
QT_MOC_LITERAL(2, 22, 0) // ""

    },
    "AbstractAlert\0dismiss\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_AbstractAlert[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void,

       0        // eod
};

void AbstractAlert::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<AbstractAlert *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->dismiss(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (AbstractAlert::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&AbstractAlert::dismiss)) {
                *result = 0;
                return;
            }
        }
    }
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject AbstractAlert::staticMetaObject = { {
    &QFrame::staticMetaObject,
    qt_meta_stringdata_AbstractAlert.data,
    qt_meta_data_AbstractAlert,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *AbstractAlert::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *AbstractAlert::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_AbstractAlert.stringdata0))
        return static_cast<void*>(this);
    return QFrame::qt_metacast(_clname);
}

int AbstractAlert::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QFrame::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void AbstractAlert::dismiss()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
struct qt_meta_stringdata_UpdateAlert_t {
    QByteArrayData data[1];
    char stringdata0[12];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_UpdateAlert_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_UpdateAlert_t qt_meta_stringdata_UpdateAlert = {
    {
QT_MOC_LITERAL(0, 0, 11) // "UpdateAlert"

    },
    "UpdateAlert"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_UpdateAlert[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void UpdateAlert::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject UpdateAlert::staticMetaObject = { {
    &AbstractAlert::staticMetaObject,
    qt_meta_stringdata_UpdateAlert.data,
    qt_meta_data_UpdateAlert,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *UpdateAlert::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *UpdateAlert::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_UpdateAlert.stringdata0))
        return static_cast<void*>(this);
    return AbstractAlert::qt_metacast(_clname);
}

int UpdateAlert::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractAlert::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_OffroadAlert_t {
    QByteArrayData data[1];
    char stringdata0[13];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_OffroadAlert_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_OffroadAlert_t qt_meta_stringdata_OffroadAlert = {
    {
QT_MOC_LITERAL(0, 0, 12) // "OffroadAlert"

    },
    "OffroadAlert"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_OffroadAlert[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void OffroadAlert::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject OffroadAlert::staticMetaObject = { {
    &AbstractAlert::staticMetaObject,
    qt_meta_stringdata_OffroadAlert.data,
    qt_meta_data_OffroadAlert,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *OffroadAlert::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *OffroadAlert::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_OffroadAlert.stringdata0))
        return static_cast<void*>(this);
    return AbstractAlert::qt_metacast(_clname);
}

int OffroadAlert::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AbstractAlert::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
