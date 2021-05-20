/*
 * Copyright (C) 2005 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Templated list class.  Normally we'd use STL, but we don't have that.
// This class mimics STL's interfaces.
//
// Objects are copied into the list with the '=' operator or with copy-
// construction, so if the compiler's auto-generated versions won't work for
// you, define your own.
//
// The only class you want to use from here is "List".
//
#ifndef _LIBS_UTILS_LIST_H
#define _LIBS_UTILS_LIST_H

#include <stddef.h>
#include <stdint.h>

namespace android {

/*
 * Doubly-linked list.  Instantiate with "List<MyClass> myList".
 *
 * Objects added to the list are copied using the assignment operator,
 * so this must be defined.
 */
template<typename T> 
class List 
{
protected:
    /*
     * One element in the list.
     */
    class _Node {
    public:
        explicit _Node(const T& val) : mVal(val) {}
        ~_Node() {}
        inline T& getRef() { return mVal; }
        inline const T& getRef() const { return mVal; }
        inline _Node* getPrev() const { return mpPrev; }
        inline _Node* getNext() const { return mpNext; }
        inline void setVal(const T& val) { mVal = val; }
        inline void setPrev(_Node* ptr) { mpPrev = ptr; }
        inline void setNext(_Node* ptr) { mpNext = ptr; }
    private:
        friend class List;
        friend class _ListIterator;
        T           mVal;
        _Node*      mpPrev;
        _Node*      mpNext;
    };

    /*
     * Iterator for walking through the list.
     */
    
    template <typename TYPE>
    struct CONST_ITERATOR {
        typedef _Node const * NodePtr;
        typedef const TYPE Type;
    };
    
    template <typename TYPE>
    struct NON_CONST_ITERATOR {
        typedef _Node* NodePtr;
        typedef TYPE Type;
    };
    
    template<
        typename U,
        template <class> class Constness
    > 
    class _ListIterator {
        typedef _ListIterator<U, Constness>     _Iter;
        typedef typename Constness<U>::NodePtr  _NodePtr;
        typedef typename Constness<U>::Type     _Type;

        explicit _ListIterator(_NodePtr ptr) : mpNode(ptr) {}

    public:
        _ListIterator() {}
        _ListIterator(const _Iter& rhs) : mpNode(rhs.mpNode) {}
        ~_ListIterator() {}
        
        // this will handle conversions from iterator to const_iterator
        // (and also all convertible iterators)
        // Here, in this implementation, the iterators can be converted
        // if the nodes can be converted
        template<typename V> explicit 
        _ListIterator(const V& rhs) : mpNode(rhs.mpNode) {}
        

        /*
         * Dereference operator.  Used to get at the juicy insides.
         */
        _Type& operator*() const { return mpNode->getRef(); }
        _Type* operator->() const { return &(mpNode->getRef()); }

        /*
         * Iterator comparison.
         */
        inline bool operator==(const _Iter& right) const { 
            return mpNode == right.mpNode; }
        
        inline bool operator!=(const _Iter& right) const { 
            return mpNode != right.mpNode; }

        /*
         * handle comparisons between iterator and const_iterator
         */
        template<typename OTHER>
        inline bool operator==(const OTHER& right) const { 
            return mpNode == right.mpNode; }
        
        template<typename OTHER>
        inline bool operator!=(const OTHER& right) const { 
            return mpNode != right.mpNode; }

        /*
         * Incr/decr, used to move through the list.
         */
        inline _Iter& operator++() {     // pre-increment
            mpNode = mpNode->getNext();
            return *this;
        }
        const _Iter operator++(int) {    // post-increment
            _Iter tmp(*this);
            mpNode = mpNode->getNext();
            return tmp;
        }
        inline _Iter& operator--() {     // pre-increment
            mpNode = mpNode->getPrev();
            return *this;
        }
        const _Iter operator--(int) {   // post-increment
            _Iter tmp(*this);
            mpNode = mpNode->getPrev();
            return tmp;
        }

        inline _NodePtr getNode() const { return mpNode; }

        _NodePtr mpNode;    /* should be private, but older gcc fails */
    private:
        friend class List;
    };

public:
    List() {
        prep();
    }
    List(const List<T>& src) {      // copy-constructor
        prep();
        insert(begin(), src.begin(), src.end());
    }
    virtual ~List() {
        clear();
        delete[] (unsigned char*) mpMiddle;
    }

    typedef _ListIterator<T, NON_CONST_ITERATOR> iterator;
    typedef _ListIterator<T, CONST_ITERATOR> const_iterator;

    List<T>& operator=(const List<T>& right);

    /* returns true if the list is empty */
    inline bool empty() const { return mpMiddle->getNext() == mpMiddle; }

    /* return #of elements in list */
    size_t size() const {
        return size_t(distance(begin(), end()));
    }

    /*
     * Return the first element or one past the last element.  The
     * _Node* we're returning is converted to an "iterator" by a
     * constructor in _ListIterator.
     */
    inline iterator begin() { 
        return iterator(mpMiddle->getNext()); 
    }
    inline const_iterator begin() const { 
        return const_iterator(const_cast<_Node const*>(mpMiddle->getNext())); 
    }
    inline iterator end() { 
        return iterator(mpMiddle); 
    }
    inline const_iterator end() const { 
        return const_iterator(const_cast<_Node const*>(mpMiddle)); 
    }

    /* add the object to the head or tail of the list */
    void push_front(const T& val) { insert(begin(), val); }
    void push_back(const T& val) { insert(end(), val); }

    /* insert before the current node; returns iterator at new node */
    iterator insert(iterator posn, const T& val) 
    {
        _Node* newNode = new _Node(val);        // alloc & copy-construct
        newNode->setNext(posn.getNode());
        newNode->setPrev(posn.getNode()->getPrev());
        posn.getNode()->getPrev()->setNext(newNode);
        posn.getNode()->setPrev(newNode);
        return iterator(newNode);
    }

    /* insert a range of elements before the current node */
    void insert(iterator posn, const_iterator first, const_iterator last) {
        for ( ; first != last; ++first)
            insert(posn, *first);
    }

    /* remove one entry; returns iterator at next node */
    iterator erase(iterator posn) {
        _Node* pNext = posn.getNode()->getNext();
        _Node* pPrev = posn.getNode()->getPrev();
        pPrev->setNext(pNext);
        pNext->setPrev(pPrev);
        delete posn.getNode();
        return iterator(pNext);
    }

    /* remove a range of elements */
    iterator erase(iterator first, iterator last) {
        while (first != last)
            erase(first++);     // don't erase than incr later!
        return iterator(last);
    }

    /* remove all contents of the list */
    void clear() {
        _Node* pCurrent = mpMiddle->getNext();
        _Node* pNext;

        while (pCurrent != mpMiddle) {
            pNext = pCurrent->getNext();
            delete pCurrent;
            pCurrent = pNext;
        }
        mpMiddle->setPrev(mpMiddle);
        mpMiddle->setNext(mpMiddle);
    }

    /*
     * Measure the distance between two iterators.  On exist, "first"
     * will be equal to "last".  The iterators must refer to the same
     * list.
     *
     * FIXME: This is actually a generic iterator function. It should be a 
     * template function at the top-level with specializations for things like
     * vector<>, which can just do pointer math). Here we limit it to
     * _ListIterator of the same type but different constness.
     */
    template<
        typename U,
        template <class> class CL,
        template <class> class CR
    > 
    ptrdiff_t distance(
            _ListIterator<U, CL> first, _ListIterator<U, CR> last) const 
    {
        ptrdiff_t count = 0;
        while (first != last) {
            ++first;
            ++count;
        }
        return count;
    }

private:
    /*
     * I want a _Node but don't need it to hold valid data.  More
     * to the point, I don't want T's constructor to fire, since it
     * might have side-effects or require arguments.  So, we do this
     * slightly uncouth storage alloc.
     */
    void prep() {
        mpMiddle = (_Node*) new unsigned char[sizeof(_Node)];
        mpMiddle->setPrev(mpMiddle);
        mpMiddle->setNext(mpMiddle);
    }

    /*
     * This node plays the role of "pointer to head" and "pointer to tail".
     * It sits in the middle of a circular list of nodes.  The iterator
     * runs around the circle until it encounters this one.
     */
    _Node*      mpMiddle;
};

/*
 * Assignment operator.
 *
 * The simplest way to do this would be to clear out the target list and
 * fill it with the source.  However, we can speed things along by
 * re-using existing elements.
 */
template<class T>
List<T>& List<T>::operator=(const List<T>& right)
{
    if (this == &right)
        return *this;       // self-assignment
    iterator firstDst = begin();
    iterator lastDst = end();
    const_iterator firstSrc = right.begin();
    const_iterator lastSrc = right.end();
    while (firstSrc != lastSrc && firstDst != lastDst)
        *firstDst++ = *firstSrc++;
    if (firstSrc == lastSrc)        // ran out of elements in source?
        erase(firstDst, lastDst);   // yes, erase any extras
    else
        insert(lastDst, firstSrc, lastSrc);     // copy remaining over
    return *this;
}

}; // namespace android

#endif // _LIBS_UTILS_LIST_H
