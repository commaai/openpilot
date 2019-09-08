#ifndef GRAPHBUILDER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define GRAPHBUILDER_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/mark.h"
#include <string>

namespace YAML {
class Parser;

// GraphBuilderInterface
// . Abstraction of node creation
// . pParentNode is always NULL or the return value of one of the NewXXX()
//   functions.
class GraphBuilderInterface {
 public:
  // Create and return a new node with a null value.
  virtual void *NewNull(const Mark &mark, void *pParentNode) = 0;

  // Create and return a new node with the given tag and value.
  virtual void *NewScalar(const Mark &mark, const std::string &tag,
                          void *pParentNode, const std::string &value) = 0;

  // Create and return a new sequence node
  virtual void *NewSequence(const Mark &mark, const std::string &tag,
                            void *pParentNode) = 0;

  // Add pNode to pSequence.  pNode was created with one of the NewXxx()
  // functions and pSequence with NewSequence().
  virtual void AppendToSequence(void *pSequence, void *pNode) = 0;

  // Note that no moew entries will be added to pSequence
  virtual void SequenceComplete(void *pSequence) { (void)pSequence; }

  // Create and return a new map node
  virtual void *NewMap(const Mark &mark, const std::string &tag,
                       void *pParentNode) = 0;

  // Add the pKeyNode => pValueNode mapping to pMap.  pKeyNode and pValueNode
  // were created with one of the NewXxx() methods and pMap with NewMap().
  virtual void AssignInMap(void *pMap, void *pKeyNode, void *pValueNode) = 0;

  // Note that no more assignments will be made in pMap
  virtual void MapComplete(void *pMap) { (void)pMap; }

  // Return the node that should be used in place of an alias referencing
  // pNode (pNode by default)
  virtual void *AnchorReference(const Mark &mark, void *pNode) {
    (void)mark;
    return pNode;
  }
};

// Typesafe wrapper for GraphBuilderInterface.  Assumes that Impl defines
// Node, Sequence, and Map types.  Sequence and Map must derive from Node
// (unless Node is defined as void).  Impl must also implement function with
// all of the same names as the virtual functions in GraphBuilderInterface
// -- including the ones with default implementations -- but with the
// prototypes changed to accept an explicit Node*, Sequence*, or Map* where
// appropriate.
template <class Impl>
class GraphBuilder : public GraphBuilderInterface {
 public:
  typedef typename Impl::Node Node;
  typedef typename Impl::Sequence Sequence;
  typedef typename Impl::Map Map;

  GraphBuilder(Impl &impl) : m_impl(impl) {
    Map *pMap = NULL;
    Sequence *pSeq = NULL;
    Node *pNode = NULL;

    // Type consistency checks
    pNode = pMap;
    pNode = pSeq;
  }

  GraphBuilderInterface &AsBuilderInterface() { return *this; }

  virtual void *NewNull(const Mark &mark, void *pParentNode) {
    return CheckType<Node>(m_impl.NewNull(mark, AsNode(pParentNode)));
  }

  virtual void *NewScalar(const Mark &mark, const std::string &tag,
                          void *pParentNode, const std::string &value) {
    return CheckType<Node>(
        m_impl.NewScalar(mark, tag, AsNode(pParentNode), value));
  }

  virtual void *NewSequence(const Mark &mark, const std::string &tag,
                            void *pParentNode) {
    return CheckType<Sequence>(
        m_impl.NewSequence(mark, tag, AsNode(pParentNode)));
  }
  virtual void AppendToSequence(void *pSequence, void *pNode) {
    m_impl.AppendToSequence(AsSequence(pSequence), AsNode(pNode));
  }
  virtual void SequenceComplete(void *pSequence) {
    m_impl.SequenceComplete(AsSequence(pSequence));
  }

  virtual void *NewMap(const Mark &mark, const std::string &tag,
                       void *pParentNode) {
    return CheckType<Map>(m_impl.NewMap(mark, tag, AsNode(pParentNode)));
  }
  virtual void AssignInMap(void *pMap, void *pKeyNode, void *pValueNode) {
    m_impl.AssignInMap(AsMap(pMap), AsNode(pKeyNode), AsNode(pValueNode));
  }
  virtual void MapComplete(void *pMap) { m_impl.MapComplete(AsMap(pMap)); }

  virtual void *AnchorReference(const Mark &mark, void *pNode) {
    return CheckType<Node>(m_impl.AnchorReference(mark, AsNode(pNode)));
  }

 private:
  Impl &m_impl;

  // Static check for pointer to T
  template <class T, class U>
  static T *CheckType(U *p) {
    return p;
  }

  static Node *AsNode(void *pNode) { return static_cast<Node *>(pNode); }
  static Sequence *AsSequence(void *pSeq) {
    return static_cast<Sequence *>(pSeq);
  }
  static Map *AsMap(void *pMap) { return static_cast<Map *>(pMap); }
};

void *BuildGraphOfNextDocument(Parser &parser,
                               GraphBuilderInterface &graphBuilder);

template <class Impl>
typename Impl::Node *BuildGraphOfNextDocument(Parser &parser, Impl &impl) {
  GraphBuilder<Impl> graphBuilder(impl);
  return static_cast<typename Impl::Node *>(
      BuildGraphOfNextDocument(parser, graphBuilder));
}
}

#endif  // GRAPHBUILDER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
